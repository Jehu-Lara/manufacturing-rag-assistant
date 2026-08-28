# ADR-007: Hugging Face PRO Docker Space deployment

## Status

Accepted on 2026-08-28. This supersedes ADR-005 only for the current deployment
target; ADR-005 remains the historical record of the Oracle-era decision.

## Context

Oracle Cloud Always Free Ampere A1 capacity remained unavailable in the selected
region. The project owner chose a second public Hugging Face Space for the live
interactive application. The existing Static Space remains a portfolio showcase
and is not the runtime.

The deployment must be reproducible, manually promoted, keyless, recoverable by
SHA, and small enough for CPU Basic's 2 vCPU and 16 GB memory limit. Storage is
ephemeral. The external LLM provider remains a runtime dependency.

## Decision

### Image and runtime

- Use `python:3.11-slim-bookworm` pinned by digest.
- Install `ca-certificates`, nginx, and tini in one system layer.
- Install the entire Python lock in one resolver transaction with the PyTorch
  CPU wheel index, then run `pip check` during the build.
- Build and run application artifacts as UID 1000. Source enters the image with
  `COPY --chown`; the multi-gigabyte model cache and index are never followed by
  a recursive ownership rewrite.
- Cache `BAAI/bge-m3` under `/home/user/.cache/huggingface`, pin revision
  `5617a9f61b028005a4858fdac845db406aefb181`, bake the corpus/index into the
  image, and set `HF_HUB_OFFLINE=1` for runtime.
- Use tini as the entrypoint. `start.sh` supervises nginx, FastAPI, and Streamlit,
  forwards termination signals, reaps direct children, and stops the whole
  container when any child exits.

The lock remains exact-version-only. Adding dependency hashes is intentionally a
separate supply-chain task.

### Public surface

```text
Internet -> nginx :7860 -> Streamlit :8501 -> FastAPI :8000
                          loopback          loopback
```

nginx is the only process bound publicly. `/`, `/health`, and `/ready` are the
only public routes. `/query`, `/docs`, `/docs/`, and `/openapi.json` return 404
at nginx. Streamlit calls `/query` directly over loopback and still supplies
`API_KEY` as defense in depth. The 20 requests/minute limiter is a shared demo
budget because every UI request reaches FastAPI from the Streamlit process.

nginx logs to stdout/stderr, hides its version, and preserves WebSocket support.
No cross-origin isolation headers are added without a demonstrated need.

### Space configuration

After explicit authorization, create the public Docker Space
`JehuLara/manufacturing-rag-assistant-live` on CPU Basic. The deployment card is
stored at `deploy/hf-space/README.md` and becomes the staging root `README.md`.
It declares `sdk: docker`, `app_port: 7860`, `license: mit`, CPU Basic, BGE-M3,
and technical tags.

Runtime secrets in Hugging Face Settings:

- `GROQ_API_KEY`
- `OPENAI_API_KEY`
- `API_KEY`

Runtime variables:

- `API_BASE_URL=http://127.0.0.1:8000`
- `LLM_PROVIDER=groq`
- `REFUSAL_COSINE_THRESHOLD=0.5999`
- `RATE_LIMIT_PER_MINUTE=20`
- `LOG_LEVEL=INFO`

Secrets and runtime variables are never Docker build arguments.

### Manual keyless publication

Configure a Hugging Face repo Trusted Publisher with these exact claims:

- repository: `Jehu-Lara/manufacturing-rag-assistant`
- branch: `master`
- workflow: `deploy-hf-space.yml`
- resource: `spaces/JehuLara/manufacturing-rag-assistant-live`

Create the GitHub Environment `hf-live` and restrict it to `master`. A required
reviewer is not configured until a second trusted person exists; the workflow's
manual dispatch and exact confirmation remain mandatory.

`.github/workflows/deploy-hf-space.yml` accepts a 40-character `git_sha` and
requires `confirm=DEPLOY`. It verifies that the commit is an ancestor of
`origin/master` and has a successful completed `CI` run. It uses least-privilege
permissions (`contents: read`, `actions: read`, `id-token: write`), serial
concurrency, checkout without persisted credentials, and actions pinned by SHA.
The fixed `huggingface_hub` CLI obtains a short-lived repo-scoped token through
`HF_OIDC_RESOURCE`; no long-lived `HF_TOKEN` is stored.

The upload staging allowlist is:

- `src/`
- `corpus/`
- `Dockerfile`
- `requirements-lock.txt`
- `nginx.conf`
- `start.sh`
- `LICENSE`
- the Space card as root `README.md`
- generated `DEPLOYED_SHA` and `DEPLOYED_AT`

The workflow mirrors staging with `hf upload ... --repo-type space --delete '*'`.
Deletion is intentional and bounded by the allowlist, Environment, exact manual
confirmation, serialized deployment, and green selected SHA.

Rollback is the same workflow with the last known-good SHA. There are no database
migrations or mutable deployment transactions to reverse; corpus and indexes are
immutable image content.

## Promotion criteria

Before deployment:

1. Tests, lint, type checks, eval integrity, and dependency integrity pass.
2. A new image builds from the hardened Dockerfile; UID 1000 and `pip check` are
   confirmed, with no CUDA/NVIDIA packages.
3. Image size, idle RSS, and peak RSS are measured; provisional peak limit is
   12.8 GB.
4. `/health` is 200, `/ready` reaches 200 within 180 seconds, and `/` loads.
5. External `/query`, `/docs`, `/docs/`, and `/openapi.json` return 404.
6. Internal requests prove 401 without the key, 422 for invalid payload, and 429
   after the shared quota.
7. English and Spanish answer/refusal paths include citations and finish within
   60 seconds.
8. SIGTERM and a child-process failure shut down the entire container cleanly.
9. Logs and layers contain no keys, authorization headers, or sensitive content.

After upload:

1. Wait for `RUNNING` and verify `DEPLOYED_SHA` against the selected commit.
2. Repeat health, readiness, UI, bilingual, and public-route checks.
3. Factory Restart the Space and repeat readiness.
4. Measure three cold starts before considering a startup-timeout override.
5. Only then call Phase 3 operational and update the Static Space's destination.

## Residual risks

- Cold starts and automatic sleep may delay the first request.
- CPU Basic may exceed the 16 GB memory ceiling or take too long to build.
- Groq/OpenAI availability remains outside this repository's control.
- `--delete '*'` intentionally removes remote files absent from staging.
- Trusted Publisher, Environment, Space secrets/variables, real CPU Basic
  measurements, and production verification are external manual gates.
