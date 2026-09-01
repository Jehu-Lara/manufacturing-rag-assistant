# ADR-009: Phase 3 bilingual refusal — grounded-review gate band

## Status

**Proposed 2026-08-31.** Written before the `grounded_review` policy is
implemented. It becomes **Accepted** only if a real `generation_eval` run (single
provider, no rate-limit-driven fallback) on the frozen 48-question holdout clears
every Phase 3 hard-stop and the project owner signs off on the observed
LLM-call / latency cost. Until then the shipped default stays
`REFUSAL_POLICY=binary` at `REFUSAL_COSINE_THRESHOLD=0.5999` and this ADR is a
plan of record, not a decision in force.

Phase 3A (startup integrity guards, this ADR's §"Startup fail-fast") is
implemented and in force as of commit `273cf1e`; Phase 3A.1 hardens the holdout
guard and adds this ADR; Phase 3A.2 makes the holdout guard pair-aware. Phase 3B
(the policy code, prompts, HTTP fields and UI) is implemented behind the
`binary` default — it changes no behaviour until `REFUSAL_POLICY=grounded_review`
is set, and the default flip stays gated on the Phase 3C holdout run. Phase 3C
(§"Measurement" below) adds the runner, guards and provider hooks needed for
that run; it does not itself flip anything.

## Context

`contextual-v1` / `off` shipped (ADR-008) and cleared the Spanish Recall@5 gate,
but left the reported bilingual false-refusals unfixed: the expected chunk *is*
retrieved for `r001` / `r002`, and the `0.5999` cosine gate refuses anyway.
ADR-008 explicitly routed those 15 gate-over-refusals to "a separate Phase 3
gate-calibration plan; `REFUSAL_COSINE_THRESHOLD` may only change there."

Measured on the live `contextual-v1` / `off` index (regression report
`eval/reports/regression_eval_v1.1.0__contextual-v1__off.md`, reproduced locally
by `gate_score_guard`):

| id   | lang | should answer | top1 semantic | current gate |
|------|------|---------------|---------------|--------------|
| r001 | en   | yes           | 0.5642        | REFUSE       |
| r002 | es   | yes           | 0.5656        | REFUSE       |
| r018 | en   | no            | 0.5630        | REFUSE       |
| r019 | es   | no            | 0.5001        | REFUSE       |
| r020 | en   | no            | 0.5420        | REFUSE       |

`r001` / `r002` sit ~0.036 below the `0.5999` cutoff. Lowering the single
threshold to reach them also lets `r018`–`r020` (controls that must refuse)
through — the same trade ADR-008 rejected. `expansion_mode=semantic` reaches
`r001` / `r002` but flips all three controls to "answer" (ADR-008, rejected).

## Decision (proposed)

### 1. Startup fail-fast on index incoherence — in force (Phase 3A)

Phase 2's close claimed the eval runners reject a drifted index, but the serving
process did not: `app.lifespan()` never called `index_manifest.verify()`, and
`/ready` only checked that Chroma opened a collection. A container could serve a
manifest, Chroma collection and BM25 index that disagree on profile, count or
model.

`index_manifest.verify()` now also checks embedding model + revision,
`chunk_count` against `chunks.jsonl`, and an optional `expected_profile`, and
returns the manifest. `ChromaVectorStore.validate_collection()` and
`Bm25LexicalIndex.validate()` are new startup guards. `app.lifespan()` runs, in
order: resolve profile → `verify(expected_profile=...)` → `load_chunks()` →
construct embedder / stores → `validate_collection()` → `bm25.validate()` →
only then build the retriever and use case. **An incoherent index fails to boot**
rather than surfacing as a late `/query` 500. CI builds the index before the
test step, and tests that start the app skip (not error) when no index is on
disk (`tests/conftest.py::built_retrieval_index_present`).

### 2. `REFUSAL_REVIEW_FLOOR = 0.5500`, pre-registered (Phase 3B)

A three-band gate replaces the binary one when `REFUSAL_POLICY=grounded_review`:

- `score < 0.5500` (or missing) → `hard_refuse`, exactly today's behaviour, zero
  LLM calls.
- `0.5500 <= score < 0.5999` → `grounded_review`: one *logical* structured LLM
  call whose answer is accepted only if every cited chunk is in the top-5 and
  carries an exact verbatim quote from the raw `chunk_text`; otherwise it
  degrades to the canonical refusal. "One logical call" is the gate's contract,
  not a network guarantee — the LLM adapter may still perform physical provider
  retries, JSON repair, or a primary→fallback provider switch underneath. In this
  band a self-refusal (`refused: true`) always yields the canonical refusal
  string, never the model's own text (`decision_reason` `llm_self_refusal`);
  `binary` / `confident` keep the legacy behaviour of passing the model's
  refusal text through.
- `score >= 0.5999` → `confident`, today's prompt and path unchanged **except**
  for the empty-answer guard below.

**One small deviation from "`binary`/`confident` unchanged":** in every band, a
non-refused answer whose body is empty or whitespace-only is now degraded to the
canonical refusal (`decision_reason` `empty_answer`) instead of being returned as
a blank `refused: false` answer. This is a strict safety improvement — a blank
answer was never useful — and it is the only behavioural change `binary` and the
`confident` band see under Phase 3B. It is covered by `test_query_use_case.py`
(`test_confident_empty_answer_with_valid_citations_downgrades` and the grey-band
equivalent).

`0.5999` keeps its exact current meaning (high-confidence boundary) and is not
touched. The floor `0.5500` is fixed **now, before the holdout is authored**, by
this rule:

```
min(r001, r002) top1_semantic on contextual-v1/off = 0.5642
required margin below the known-good cases                = 0.0100
=> floor must be <= 0.5542; the highest 0.01 step that fits = 0.5500
```

This keeps `r001` / `r002` in the review band with margin >= 0.0142 and leaves
`r019` (0.5001) and `r020` (0.5420) hard-refused for free. `r018` (0.5630) enters
the review band and **must be refused by the grounded review** — that is a
Phase 3 hard-stop, not an accepted risk. `gate_score_guard` (CI) fails if
`r001` / `r002` leave `[0.5500, 0.5999)` on the freshly built index; the
response is to re-measure, never to retune the floor toward individual cases.

### 3. Additive HTTP fields + legacy snapshot projection (Phase 3B)

`QueryResponse` gains `review_floor: Optional[float]` and
`gate_band: "hard_refuse" | "grounded_review" | "confident"`. No existing field
is removed, renamed or redefined. The Phase 0 contract snapshot
(`tests/snapshots/http_contract_phase0.json`) is **not** rewritten; the snapshot
test projects the frozen legacy key set before comparing and asserts the two new
keys separately. The internal `decision_reason` (why a grey answer was accepted
or downgraded) is logged and passed to the evaluator but never serialized in the
HTTP response.

The Phase 0 snapshot froze the legacy fields *and their values*; it does not
prohibit backward-compatible extension. This ADR is the record of that reading.

### 4. Holdout frozen before Phase 3B measurement

`eval/gate_holdout_v1.0.0.json` — 48 questions, 12 answerable + 12 unanswerable
intents each paired EN/ES (24/24 by class, 24/24 by language), no paraphrases of
`eval_set` v1.1.0 or `regression_queries.json`. `gate_holdout_integrity.verify()`
enforces `status == "frozen"`, the 48-count, unique ids, the class/language
balance, per-class required fields, and that every `expected_chunk_ids` entry
exists in `chunks.jsonl`. The file ships as an **empty draft**; CI's holdout step
is **red by design** until the owner authors and freezes it. A green integrity
check over `[]` would be a false guarantee.

Chunk ids in the holdout are derived from the corpus and verified against
`chunks.jsonl` — never invented. Whoever implements the grey prompt must not
adjust its rules after seeing holdout results.

The `gate_holdout_integrity` guard additionally requires `eval_set` and
`regression_queries` to be present **and pass their own hash check** before the
holdout is de-duplicated against them (a missing or tampered source could
otherwise silently narrow the comparison), rejects two holdout questions that
are identical after NFKC+casefold+whitespace, requires the EN and ES halves of
an answerable pair to target the same `expected_chunk_ids`, and writes
atomically.

### 5. Measurement — the Phase 3C causal run (not a decision, an instrument)

Before the paid run, `gate_holdout_profile` buckets all 48 frozen questions by
band on `contextual-v1/off` and **fails unless each EN/ES × answerable/
unanswerable cell has ≥ 3 questions in `[0.5500, 0.5999)`**. A holdout that
barely enters the grey band cannot tell us whether `grounded_review` helps; the
fix is another draft with more borderline questions, never a lower floor.

`gate_generation_eval` then measures `binary` vs `grounded_review` without
confounds:

- Retrieval is snapshotted **once** per question and replayed identically to a
  `binary` and a `grounded_review` `QueryUseCase` — the two runs differ only in
  policy.
- `index_profile`, `expansion_mode`, floor, threshold and provider are pinned in
  the runner, not read from the environment.
- The `confident` band produces a byte-identical LLM call under both policies, so
  it is issued once and shared **within a repeat**; a fresh cache per repeat
  means a response is never reused across repeats. Grey-band prompts differ and
  are always separate calls. `hard_refuse` calls nothing.
- `--provider {groq|openai}` is **required** and must match `LLM_PROVIDER`.
  Provider fallback is forced **off** (`allow_provider_fallback=False`) so a
  rate-limit fallover can't swap the model mid-comparison. A content-free
  `trace_hook` records **every physical provider round trip** — a `physical_attempt`
  before each call and one `physical_request`/`physical_failed` after, so 429s,
  schema fallbacks and network failures all count toward cost — plus tokens,
  finish reason, per-call latency, rate-limit, repair and schema-fallback
  counts. Never a prompt, answer or key.
- Reported latency is **modelled, not question wall-clock**: generation p50/p95
  over physical LLM calls, and a separate retrieval p50/p95 timed once over the
  live index. Replayed retrieval (~0 ms) and within-repeat confident-call reuse
  are excluded by construction.
- A grey-band coverage preflight (same rule as `gate_holdout_profile`, ≥3 per
  EN/ES × class cell) runs on the snapshot and aborts the paid run if the
  holdout barely touches the band.
- 3 full holdout repeats + 3 repeats of the `r001/r002/r018–r020` canary.
  Artifacts land in a **write-once, atomic** directory (`<id>.partial/` then
  renamed): `run_manifest.json`, `retrieval.jsonl`, `outcomes.jsonl`,
  `comparison.md`, `blind_checklist.csv`, `arm_map.sealed.json`,
  `checksums.txt`.
- The blind checklist is one row **per attempt** (repeat × arm × question),
  labelled `arm-A`/`arm-B` with the policy mapping sealed away; it carries the
  generated answer, the cited chunk ids + their text, and the expected
  answer/chunks, and it **includes every unanswerable question that was
  answered** for a safety grade. `--import-verdicts <run_dir>` reads the graded
  file back, resolves the citation/faithfulness and unsafe-answer gates, and
  rewrites `comparison.md`.

Gates (all must hold): zero errors / provider or schema fallbacks / 429s;
grounded correct-refusal ≥ binary globally and per language; grounded
false-refusal not worse per language and strictly better globally; `r001`/`r002`
answered **and citing the expected chunk** 3/3, `r018`–`r020` refused 3/3 under
`grounded_review`; no unanswerable answered unsafely (blind `safe_pass`);
conditional citation/faithfulness ≥ 0.90 (blind). **These gates are advisory
input to a human decision — the runner never flips the default.**

### 6. Closure order

1. `ruff`, `mypy --strict`, the full test suite and every integrity guard green
   (the `gate_holdout` step stays red until the holdout is frozen — that is the
   one intended exception, and it must go green as part of this step, not be
   waived).
2. Owner authors + freezes the holdout; `gate_holdout_profile` passes.
3. Owner gate: an explicit decision to spend, then the paid `gate_generation_eval`
   run and the blind review.
4. **Only if every gate passes:** flip the default to `grounded_review`, move this
   ADR to `Accepted`, update `SPEC.md` / `CLAUDE.md`. Otherwise `binary` stays
   and there is no reindex, push, merge or deploy.

## Alternatives considered

- **Lower the single `0.5999` threshold** — rejected, same reason as ADR-008: it
  trades false-refusals for false-answers on `r018`–`r020` and cannot help a
  chunk outside top-5.
- **Per-language thresholds** — rejected. The Spanish eval subset history
  (n=7 → n=47) shows the project's caution about per-language tuning; a
  content-verification step generalizes better than a language-indexed constant.
- **`expansion_mode=semantic` in production** — rejected in ADR-008 (controls
  0/3).
- **Fuzzy / token-overlap quote matching** — rejected. Verbatim substring after
  NFKC + whitespace normalization is the integrity guarantee; a fuzzy match
  reintroduces the "plausible but unsupported" failure mode the citation design
  exists to prevent. Whitespace/newline collapse is the only tolerance.
- **A model_validator on `Settings`** — rejected. `Settings` stays an inert data
  holder (its docstring, and ~15 tests that construct it directly). Floor /
  threshold validation lives in `load_settings()` and `RefusalPolicy.__init__`.

## Consequences

- Every grey-band query costs one full generation call where today it costs an
  instant gate refusal. `r018`–`r020`, the ~8 gate-over-refusals, and an unknown
  share of the `eval_set` unanswerable subset all move into that band. The
  Phase 3B evaluation must report the LLM-call count, p50/p95 latency, and the
  per-band traffic share; the default flip is gated on the owner accepting those
  numbers.
- `grounded_review` safety rests partly on the LLM reliably refusing a grey-band
  question whose retrieved context does not actually answer it. The verbatim
  quote check is procedural evidence of provenance, not proof of entailment; the
  holdout's human review is the backstop.
- Cosine scores can drift slightly between Windows and Linux / numeric library
  versions. `gate_score_guard` checks the *band*, never an exact value.
- Startup is stricter: a mis-built or partially-swapped index now blocks boot.
  This is the intended trade — a container that cannot serve a coherent index
  should fail its readiness probe, not answer from a stale one.

## Rollback

Operational: `REFUSAL_POLICY=binary` and `REFUSAL_COSINE_THRESHOLD=0.5999`
(both `.env` values). No reindex. `binary` reproduces today's behaviour exactly.

Reopen Phase 2 (rebuild `contextual-v1` and re-measure) only if the manifest,
Chroma collection and BM25 index disagree physically, or the Linux index does
not place `r001` / `r002` in `[0.5500, 0.5999)`. Do not compensate for a failed
hard-stop by moving the floor, the threshold, or the prompt rules.

Never mutate `eval/eval_set.json` or `eval/regression_queries.json`, never
regenerate a frozen `*_v1.0.0` report, and never relax a Phase 3 hard-stop to
make `grounded_review` pass.
