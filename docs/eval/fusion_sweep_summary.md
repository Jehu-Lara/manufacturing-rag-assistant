# RRF fusion parameter sweep — result

Run: `python -m src.features.evaluation.fusion_sweep`
Full report: `eval/reports/fusion_sweep_v1.1.0__contextual-v1__off.md`
Date: 2026-09-04 · eval set v1.1.0 · live index `contextual-v1` · `expansion_mode=off`

## Why this run exists

`docs/eval/ablation_summary.md` found the lexical channel net-negative: zero
exclusive rescues, and eight questions demoted out of the top 5 that the
semantic channel alone retrieves. It named one confounder to rule out before
concluding anything about BM25 itself — RRF's rank-only fusion at `k=60`.

This run rules it **in**. The loss is a property of the fusion parameters, not
of what BM25 retrieves.

## The mechanism

RRF at `k=60` over 20 candidates scores rank 1 at `1/61 = 0.01639` and rank 20
at `1/80 = 0.01250` — a **1.31x** spread across the entire ranking. A chunk
found by *both* channels scores the sum of two such terms: roughly **2x**.

So membership in both rankings is worth more than any rank difference within
one ranking. A mediocre chunk at semantic #11 that BM25 also liked outranks the
gold chunk at semantic #1 that BM25 missed. Fusion degenerates into "how many
channels found it", with rank as a tiebreak.

The per-question evidence, for the eight questions the shipped fusion loses:

| question | gold's semantic rank | gold's BM25 rank |
|---|---|---|
| q014 | 1 | — |
| q017 | 2 | — |
| q026 | 1 | 19 |
| q050 | 1 | — |
| q051 | 1 | — |
| q066 | 2 | — |
| q075 | 1 | — |
| q083 | 3 | — |

Five at semantic rank 1, two at rank 2, one at rank 3 — and all eight land
outside the fused top 5. That is a fusion result, not a retrieval failure.

A separate check of what sits above them: of 62 chunks ranked above the gold
chunk across those eight questions, only **4** were BM25-only (absent from the
semantic top-40). The displacers are overwhelmingly chunks both channels found
at mediocre semantic rank — which is the both-channels bonus, not a weak
lexical rank-1 sneaking in.

## Results

| rule | Recall@5 | EN | ES | MRR | new misses | rescues |
|---|---|---|---|---|---|---|
| `rrf_k60` (shipped) | 0.887 | 0.917 | 0.844 | 0.721 | 0 | 0 |
| `rrf_k20` | 0.900 | 0.917 | 0.875 | 0.725 | 0 | 1 |
| `rrf_k10` | 0.938 | 0.938 | 0.938 | 0.740 | 0 | 4 |
| `rrf_k5` | 0.950 | 0.958 | 0.938 | 0.765 | 0 | 5 |
| `rrf_k1` | 0.975 | 0.979 | 0.969 | 0.769 | 0 | 7 |
| `rrf_k60_sem_x2` | 0.900 | 0.917 | 0.875 | 0.731 | 0 | 1 |
| `rrf_k60_sem_x3` | 0.900 | 0.917 | 0.875 | 0.727 | 0 | 1 |
| `rrf_k10_sem_x2` | 0.988 | 0.979 | 1.000 | 0.798 | 0 | 8 |
| `semantic_only` | 0.988 | 0.979 | 1.000 | 0.835 | 0 | 8 |

`semantic_only` reproduces the ablation's number exactly (0.988 / 0.979 / 1.000).
The two modules compute it by different routes, so that is a free cross-check
that neither is measuring a fiction.

## What it means

1. **Recall is monotone in decreasing `k`.** Flattening the rank curve less
   lets a strong semantic rank outweigh a shared-but-mediocre one. This is the
   confounder, confirmed.

2. **Weighting alone barely helps at `k=60`.** Semantic ×2 and ×3 both land at
   0.900. At that flatness the curve is so compressed that tripling one
   channel's weight still cannot beat the additive bonus for many candidates.
   Both levers have to move together: `rrf_k10_sem_x2` reaches 0.988.

3. **Fixing the fusion stops BM25 from subtracting; it does not make BM25 add.**
   `rrf_k10_sem_x2` ties `semantic_only` on recall in every language, and still
   loses on MRR (0.798 vs 0.835). On this set the best case for the lexical
   channel is "harmless".

4. **The gates do not discriminate here.** Every rule passes all three and has
   zero new misses — expected, since the thresholds were set at the current
   baseline, so anything at or above it passes. Read the gates as a floor that
   nothing violated, not as evidence any row is better.

## What must NOT happen next

**Do not pick a `k` off this table.** It was measured on the same 80 questions
it would be tuned to, over a 14-document corpus. That is precisely the
tune-to-the-measurement-set failure the project already forbids for
`REFUSAL_REVIEW_FLOOR` against the Phase 3 holdout, and `k=60` is a byte-stable
invariant for the same reason.

Changing it needs what the review floor needed: its own calibration set,
separate from a newly frozen holdout that the chosen value never touched, plus
an ADR recording the pre-registered choice. Nothing here is that.

What this run *is* good for: it tells the owner that "remove BM25" and "retune
the fusion" are both live options where before there was only the first, and
that a decision on either needs a properly separated calibration set.

## Provenance

Fusion is a pure function of two ranked id lists, so this replays it offline
over channel outputs captured once per question — exact, not approximated.
`tests/test_evaluation_fusion_sweep.py::test_baseline_rule_reproduces_the_shipped_retriever_exactly`
pins the `rrf_k60` row against the real `HybridRetriever` so the baseline can
never drift into a lookalike.

No index was written, no LLM was called, and no default changed.
