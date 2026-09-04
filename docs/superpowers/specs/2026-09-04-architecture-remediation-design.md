# Architecture Remediation & Evolution — Design Spec

**Date:** 2026-09-04
**Status:** Approved by owner (audited plan, verdict PASS / severity P2)
**Baseline audited:** `fix/audit-fixes-t1-t5@88342bf` — 547 tests green, Ruff green, mypy green, manifest coherent. PR #9 CI green, `MERGEABLE`.

## Authorization gates (binding)

These are preconditions on *execution*, not on planning. Planning (this spec + the five plans) is unblocked and done.

1. **PR #9 must be resolved by the owner** (merged or closed). Nothing here is built on top of an open PR.
2. **The owner must authorize a new branch cut from `master`** after that resolution.
3. No paid `gate_generation_eval` run, no reindex, no corpus change, no commit, no push, no merge, no deploy — each needs its own separate, explicit authorization.

## Audited findings (confirmed against the code)

| # | Finding | Evidence in repo |
|---|---|---|
| A | Positional chunk ids | `src/features/ingestion/cli.py` builds `f"{document_id}::chunk-{index:04d}"` |
| B | 15 `features → adapters` imports | `grep -rn "from src.adapters" src/features/` returns 15 lines |
| C | Evaluation is 51.2% of `src` | 3822 of 7498 LOC (51.0%) under `src/features/evaluation/` |
| D | `Settings` leaks into `LLMClientPort` | `src/domain/ports.py:39` — `generate_structured(..., settings: Settings)` |
| E | Contextual embedding policy lives inside the Chroma adapter | `src/adapters/secondary/vector/chroma_vector_store.py:14-16,56-59` |
| F | `gate_generation_eval.py` is 1143 lines | `wc -l` |
| G | Scattered env reads | `INDEX_PROFILE`/`DEPLOYED_SHA` in `index_manifest.py`, `OTEL_EXPORTER_OTLP_ENDPOINT` in `telemetry.py` — outside `load_settings()` |
| H | Repeated repo-root path chains | 16 occurrences of `parent.parent.parent[.parent]` across `src/` |
| I | Module-level mutable singleton | `src/core/telemetry.py:15` — `_configured = False` |

## Nuances carried from the audit (do not silently drop)

- **"BM25 aporta ~0" is unproven.** It produced no exclusive top-5 rescues in ES, but that is not an ablation. Bucket 5 measures it before anyone acts on it.
- **Translation is not a priority** — BGE-M3 is natively multilingual ([model card](https://huggingface.co/BAAI/bge-m3)).
- **Reranking is promising but unmeasured** — ROI and latency are unknown. Default-off experiment only ([official reranker](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/inference/reranker/README.md)).
- **Technical caches and OpenTelemetry spans do exist** — `Bm25LexicalIndex._load`, `GroqOpenAiLlmClient._sdk_client`, `get_tracer()` spans in `HybridRetriever.retrieve` and `generate_structured`.

## Byte-stable invariants (never move in this work)

- `REFUSAL_COSINE_THRESHOLD = 0.5999`
- `REFUSAL_REVIEW_FLOOR = 0.5500` (never recalibrated against the Phase 3 holdout)
- RRF `k = 60`, ties broken by ascending `chunk_id`
- `REFUSAL_POLICY` default `binary`; `expansion_mode` default `off`; served profile `contextual-v1`
- Frozen datasets (`eval/eval_set.json`, `eval/regression_queries.json`, `eval/gate_holdout_v1.0.0.json`) never re-stamped
- `eval/reports/*_v1.*` and `*__raw-v1__off.*` baselines never overwritten

## Acceptance gates for any retrieval candidate (Bucket 5)

No candidate replaces `contextual-v1/off` unless **all** hold, and even then it needs separate owner approval:

- EN Recall@5 ≥ 0.917
- ES Recall@5 ≥ 0.844
- Global Recall@5 ≥ 0.887
- Zero new misses relative to the current baseline

## Global test bar (every bucket)

RED→GREEN per batch. HTTP/port contracts, secret non-disclosure, and rollback are all covered by tests. At the end of each bucket: full `pytest -q` (547+), `ruff check src tests`, `mypy src`, the three `*_integrity --verify` commands, and CI green.

## Scope split

Five independent subsystems, one plan each. Each plan produces working, testable software on its own.

1. `2026-09-04-arch-1-http-config-hygiene.md` — structural/config/validation hygiene
2. `2026-09-04-arch-2-llm-port-decoupling.md` — `LLMClientPort` without `Settings`; adapter split
3. `2026-09-04-arch-3-index-ownership-versioning.md` — embedding-text ownership, BM25 versioning, CLI build ownership
4. `2026-09-04-arch-4-gate-eval-decomposition.md` — split `gate_generation_eval.py` behind a compatible façade
5. `2026-09-04-arch-5-retrieval-experiments.md` — ablation + reranker, default-off

## Explicitly deferred (do not build)

Translation layer, parent-window retrieval, stable v2 chunk ids, LLM judge, conversational memory, streaming, HyDE / CRAG / multi-hop. Deferred until evidence or product need demands them.

## Residual risks (accepted, unchanged by this work)

Corpus stays small (14 documents / 228 chunks). Faithfulness grading stays human. Chunk ids stay positional.
