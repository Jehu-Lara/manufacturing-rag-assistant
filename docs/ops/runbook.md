# Operations runbook

Five failure modes this deployment can actually reach, each with the signal you
would really see, how to confirm it, what to do, and how to undo it. Written for
the shipped shape: **one container** (nginx on public `:7860`, FastAPI on
loopback `:8000`, Streamlit on loopback `:8501`), the retrieval index **baked
into the image at build time**, and a Hugging Face PRO Docker Space as the
runtime.

Two properties of that shape decide most of what follows:

- **There is no runtime state to repair.** The index is image content. Anything
  wrong with it is wrong in the image, and the fix is a rebuild or a redeploy of
  a known-good image — never an in-place edit inside a running container.
- **Log lines are JSON on stdout** (`src/core/logging.py`), captured by the
  container runtime. There is no aggregator, no alerting, and no paging. Every
  "observe" step below means reading the Space's log stream. Detection is
  manual; treat that as a known limitation, not an oversight to work around.

Rollback, everywhere in this document, means the same thing: **redeploy the last
green `master` SHA** through `.github/workflows/deploy-hf-space.yml` (manual
dispatch, full 40-character SHA, exact confirmation `DEPLOY`). There is no
database, no migration, and no transactional state, so there is nothing else to
unwind.

---

## 1. Primary LLM provider (Groq) is down or rate-limiting

**Symptom.** Answers still arrive, but slowly, or intermittently fail with the
generation-error message ("A technical error occurred while generating this
answer…" / "Ocurrió un error técnico…"). This is *not* a refusal, and the UI
renders it in the error state, not the refusal state.

**Observable.** In the log stream:

- `{"event": "llm_trace", "llm_event": "rate_limited", "llm_provider": "groq", "llm_wait_seconds": …}` —
  the adapter is backing off and retrying the same provider (up to three times,
  honoring `Retry-After` when present).
- `{"event": "llm_trace", "llm_event": "provider_call_failed", "llm_provider": "groq", "llm_error_type": …}`
  followed by an `attempting structured generation` line with
  `"role": "fallback"` — the automatic failover to OpenAI fired.
- A latency rise in `{"event": "query_completed", "latency_ms": …}`.

**Diagnose.** If `llm_event` is `rate_limited` or `rate_limit_exhausted`, this is
quota, not an outage. If it is `provider_call_failed` with `llm_status_code` in
the 5xx range, it is the provider. Check Groq's status page before doing
anything.

**Act.** Normally nothing: fallback to OpenAI is automatic and already worked if
you see the fallback line. If `OPENAI_API_KEY` is not configured, the fallback
is skipped (`provider skipped because its API key is not configured`) and every
call fails — set that key in the Space's runtime secrets. To move the primary
provider deliberately, set `LLM_PROVIDER=openai` in the Space settings and
restart; this is a config change, not a deploy.

**Rollback.** Unset `LLM_PROVIDER` (back to the `groq` default) and restart.

---

## 2. Both providers are down

**Symptom.** Every question returns the generation-error message. Nothing is
answered; nothing is refused.

**Observable.** `{"event": "generation_error", "error_type": "GenerationError"}`
from the query use case on every request, preceded by
`{"event": "llm_trace", "llm_event": "generation_exhausted"}` and a
`structured generation failed on all providers` line carrying an `attempts`
summary. `/health` and `/ready` both still return 200 — retrieval is unaffected,
so this failure is invisible to a health probe by design.

**Diagnose.** Confirm the `attempts` summary names both providers. If one says
`not configured`, this is section 1 (a missing key), not a dual outage.

**Act.** There is no local generation path and none is planned — the honest
action is to wait, and to say so. If the outage is prolonged, pause the Space so
visitors see nothing rather than a stream of technical errors.

**Rollback.** None applicable; nothing was changed.

---

## 3. Index missing or unreadable at startup

**Symptom.** The container never becomes healthy. The Space shows a build/run
error, or the page loads and the Ask button stays disabled.

**Observable.** The uvicorn process exits during `lifespan` with one of:

- `FileNotFoundError: … bm25_index.json not found — run the retrieval index-build CLI first`
- a Chroma collection error from `validate_collection`
- `… no longer matches the current inputs …` from `index_manifest.verify`

`start.sh` treats any child exit as fatal, so the whole container stops. If the
API is down but the container is up, `GET /ready` returns 503 (see section 5).

**Diagnose.** The index is baked at build time by
`RUN python -m src.features.retrieval.cli` in the Dockerfile. A missing index
means that build step did not produce what startup expects — check the build log
for that step, not the runtime log.

**Act.** Rebuild the image from a green SHA and redeploy. Do not try to build an
index inside a running container: the runtime user has no network
(`HF_HUB_OFFLINE=1`) and the result would not survive a restart.

**Rollback.** Redeploy the last green SHA.

---

## 4. Index profile mismatch

**Symptom.** Startup fails specifically on a profile check, or (if it somehow
started) answers are recalled against a different embedding scheme than the one
the reports describe.

**Observable.** One of:

- `live collection index_profile is 'raw-v1', expected 'contextual-v1'` (or the
  reverse) — from `ChromaVectorStore.validate_collection`
- `index_profile stored 'raw-v1', expected 'contextual-v1'` — from
  `index_manifest.verify`
- `live collection has N rows, expected M` — the manifest and the collection
  disagree on size
- `BM25 chunk ids do not match the indexed chunks (N persisted vs M expected)` —
  the lexical and vector channels are indexing different corpora

**Diagnose.** `INDEX_PROFILE` selects the profile at **build** time; it is read
only in `src/features/retrieval/cli.run()` and is not a `Settings` field. A
mismatch means the image was built with one profile and something else expects
another — most often a partial rebuild where `chunks.jsonl` changed but the
index did not.

**Act.** Rebuild both artifacts together, in order, from a clean tree:

```bash
python -m src.features.ingestion.cli
python -m src.features.retrieval.cli
```

The default profile is `contextual-v1`. `INDEX_PROFILE=raw-v1` rebuilds the
tested rollback index; that is a deliberate decision with measured consequences
(ES Recall@5 0.844 → 0.781), not a troubleshooting step.

**Rollback.** Redeploy the last green SHA. The startup guard is the rollback
safety net: it refuses to serve an incoherent index rather than silently
answering from one.

---

## 5. `/ready` returns 503

**Symptom.** The API answers `GET /health` with 200 but `GET /ready` with
`{"status": "not_ready"}` and 503. In the UI, the Ask button is disabled and the
"index is still loading" caption is visible.

**Observable.** `{"event": "health_check_index_unreachable"}` on the `/health`
path, and a 503 body from `/ready`. Note the asymmetry, which is intentional:
`/health` is liveness only and stays 200 so its contract never changes;
`/ready` is the one that gates traffic.

**Diagnose.** `/ready` is `vector_store.ping()` — a real Chroma call. A 503
means the collection is not currently queryable. Right after a container start
this can be transient while the model and collection load; if it persists past
~3 minutes it is not warm-up, and you are actually in section 3 or 4.

**Act.** Wait out a fresh start. If it persists, read the startup log for the
errors in sections 3 and 4 and follow those. Do not restart repeatedly — a
baked index that failed to validate once will fail identically every time.

**Rollback.** Redeploy the last green SHA.

---

## What this runbook deliberately does not cover

- **Alerting.** There is none. Every detection path above is a human reading
  logs (ADR-006 — intentionally simple for a portfolio-scale app).
- **Scaling and replicas.** One container, one uvicorn worker. The rate limiter
  is per-process by design (ADR-002); there is no shared store to reconcile.
- **Data recovery.** There is no persisted user data, no database, and no
  migration. The only durable artifact is the image.
