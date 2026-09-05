# Retrieval channel ablation

Evaluation set v1.1.0, live index `contextual-v1`, `expansion_mode=off`, Recall@5 over the answerable subset.

**Nothing here changes a default.** A candidate replaces contextual-v1/off only if it clears every gate below AND introduces zero new misses AND the owner separately approves.

| arm | Recall@5 | EN | ES | MRR | new misses | rescues |
|---|---|---|---|---|---|---|
| `hybrid_word_lower` | 0.887 | 0.917 | 0.844 | 0.721 | 0 | 0 |
| `semantic_only` | 0.988 | 0.979 | 1.000 | 0.835 | 0 | 8 |

## Acceptance gates

- `hybrid_word_lower` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `semantic_only` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS

## Per-question deltas vs the current hybrid baseline

### `semantic_only`

- Recall@5 delta: +0.1000
- New misses (baseline found, candidate lost): none
- Rescues (baseline lost, candidate found): ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083']

## Does BM25 contribute anything?

- Questions the hybrid arm gets that semantic-only misses: none.
- Questions semantic-only gets that the hybrid arm misses: ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083'].

The lexical channel earns **no** exclusive rescue, and RRF fusion demotes 8 question(s) the semantic channel alone retrieves. On this evaluation set BM25 is therefore not neutral, as the audit's unproven 'BM25 aporta ~0' nuance supposed — it is net-negative.

This is a measurement, not a decision. Removing or down-weighting the lexical channel is a separate owner call, and one evaluation set of 80 answerable questions over a 14-document corpus is thin evidence for a permanent architectural change. The obvious confounder to rule out first is RRF's rank-only fusion (k=60): it weights a BM25 rank-1 hit identically however weak its score is, so a near-miss lexical match can outrank a strong semantic one.

---

## Follow-up (2026-09-04): the confounder was real, and the mechanism was not the one named above

`docs/eval/fusion_sweep_summary.md` ran the check this report asked for. The
confounder is confirmed, but the wording above pointed at the wrong path.

The claim above — a weak BM25 rank-1 outranking a strong semantic hit — accounts
for only **4 of the 62** chunks that displace the gold chunk across the eight
lost questions. The dominant mechanism is the both-channels bonus: at `k=60`
over 20 candidates the within-channel rank curve spans just `1/61..1/80`
(**1.31x**), while appearing in *both* rankings is worth roughly **2x**. So
fusion degenerates into "how many channels found it", and a shared-but-mediocre
chunk beats a semantic rank-1 that BM25 missed.

Moving both levers together (`rrf_k10_sem_x2`) recovers **0.988** Recall@5 —
identical to `semantic_only` in every language. So fixing the fusion stops BM25
from subtracting; it still does not make BM25 add, and `semantic_only` remains
ahead on MRR (0.835 vs 0.798).

**None of that licenses editing `k`.** It was measured on the same 80 questions
it would be tuned to. See that summary's "What must NOT happen next".
