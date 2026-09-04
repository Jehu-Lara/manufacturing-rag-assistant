# Architecture Remediation — Plan Index (2026-09-04)

**Spec:** [`docs/superpowers/specs/2026-09-04-architecture-remediation-design.md`](../specs/2026-09-04-architecture-remediation-design.md)

The audited plan's five `FIXES_REQUIRED` buckets are five independent subsystems, so they are five separate plans — each produces working, testable software on its own.

## Execution gates (binding, from the audit's own `BLOCKERS`)

Planning is done. **Execution has not started and must not start until both hold:**

1. The owner resolves PR #9 (merge or close).
2. The owner authorizes a new branch cut from `master` after that resolution.

Then, still individually authorized: any commit, push, merge, reindex, paid `gate_generation_eval` run, corpus change, or deploy.

## The five plans

| # | Plan | Scope | Depends on | Extra gate |
|---|---|---|---|---|
| 1 | [HTTP / config / validation hygiene](2026-09-04-arch-1-http-config-hygiene.md) | Router → HTTP adapter; `chunk_store`; `src/core/paths.py`; three env vars into `Settings`; blank-question 422; non-positive limits; `_configured` removed; AST invariants; ADR/C4 sync | — | — |
| 2 | [`LLMClientPort` decoupling](2026-09-04-arch-2-llm-port-decoupling.md) | `Settings` out of the port; provider + `SecretStr` constructor-injected; adapter split into transport / validation / tracing / failover | — | — |
| 3 | [Index ownership & versioning](2026-09-04-arch-3-index-ownership-versioning.md) | Embedding-text policy → domain; versioned BM25 payload; startup validates manifest + profile + content; CLI owns candidates → promote → manifest-last | Bucket 1 T3 (avoids a `.parent` off-by-one) | **Requires a reindex** — owner-authorized |
| 4 | [`gate_generation_eval` decomposition](2026-09-04-arch-4-gate-eval-decomposition.md) | 1143 lines → `gate_eval/{models,runner,artifacts,verdicts}` behind a compatible façade; CLI and artifact contract unchanged | Bucket 1 T3 (same reason) | No paid run |
| 5 | [Retrieval experiments](2026-09-04-arch-5-retrieval-experiments.md) | Ablation (semantic-only / current BM25 / bilingual Snowball), then `RerankerPort` + `bge-reranker-v2-m3`, all default-off | Bucket 3 T2 (`lexical_profile`) | **T3 conditional on the owner reading the T1/T2 report** |

**Suggested order:** 1 → 2 → 4 → 3 → 5. Buckets 1 and 2 are independent and safe; 4 is a pure move with a golden-artifact tripwire; 3 is deferred behind them because it is the one that forces a reindex; 5 is last because it is the only one producing new evidence rather than removing a known defect.

## What none of these plans do

Nothing here moves a byte-stable invariant (`0.5999`, `0.5500`, RRF `k=60`, `binary`, `off`, `contextual-v1`), re-stamps a frozen dataset, overwrites a baseline report, changes corpus content, or flips a default. Translation, parent-window retrieval, stable v2 chunk ids, LLM judge, memory, streaming and HyDE/CRAG/multi-hop stay deferred.

## Residual risks, unchanged

Corpus stays small (14 documents / 228 chunks). Faithfulness grading stays human. Chunk ids stay positional.
