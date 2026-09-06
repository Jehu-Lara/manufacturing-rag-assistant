# Retrieval channel ablation

Evaluation set v1.1.0, live index `contextual-v1`, `expansion_mode=off`, Recall@5 over the answerable subset.

**Nothing here changes a default.** A candidate replaces contextual-v1/off only if it clears every gate below AND introduces zero new misses AND the owner separately approves.

| arm | Recall@5 | EN | ES | MRR | new misses | rescues |
|---|---|---|---|---|---|---|
| `hybrid_word_lower` | 0.887 | 0.917 | 0.844 | 0.721 | 0 | 0 |
| `semantic_only` | 0.988 | 0.979 | 1.000 | 0.835 | 0 | 8 |
| `hybrid_snowball_bilingual` | 0.912 | 0.938 | 0.875 | 0.743 | 2 | 4 |
| `hybrid_word_lower_reranked` | 0.988 | 0.979 | 1.000 | 0.896 | 0 | 8 |

## Acceptance gates

- `hybrid_word_lower` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `semantic_only` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `hybrid_snowball_bilingual` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: FAIL (2)
- `hybrid_word_lower_reranked` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS

## Reranker latency

Wall-clock inside `rerank()` only, 20 candidates per call, one call per question. Retrieval is excluded. The percentiles cover the 79 warm calls; the first call is reported separately because it also pays the one-off model load.

- p50: 43112.9 ms
- p95: 48540.7 ms
- max: 49431.7 ms
- first (cold, includes the model load): 56504.0 ms

This is a local CPU measurement on the developer machine, not the deploy target. It bounds the shape of the cost, not the number the Space would show.


## Per-question deltas vs the current hybrid baseline

### `semantic_only`

- Recall@5 delta: +0.1000
- New misses (baseline found, candidate lost): none
- Rescues (baseline lost, candidate found): ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083']

### `hybrid_snowball_bilingual`

- Recall@5 delta: +0.0250
- New misses (baseline found, candidate lost): ['q049', 'q065']
- Rescues (baseline lost, candidate found): ['q014', 'q050', 'q051', 'q083']

### `hybrid_word_lower_reranked`

- Recall@5 delta: +0.1000
- New misses (baseline found, candidate lost): none
- Rescues (baseline lost, candidate found): ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083']

## Does BM25 contribute anything?

- Questions the hybrid arm gets that semantic-only misses: none.
- Questions semantic-only gets that the hybrid arm misses: ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083'].

The lexical channel earns **no** exclusive rescue, and RRF fusion demotes 8 question(s) the semantic channel alone retrieves. On this evaluation set BM25 is therefore not neutral, as the audit's unproven 'BM25 aporta ~0' nuance supposed — it is net-negative.

This is a measurement, not a decision. Removing or down-weighting the lexical channel is a separate owner call, and one evaluation set of 80 answerable questions over a 14-document corpus is thin evidence for a permanent architectural change. Rule out the fusion parameters first: at k=60 over 20 candidates, RRF's rank curve spans only 1/61..1/80 (1.31x) while appearing in BOTH rankings is worth roughly 2x, so a shared-but-mediocre chunk outranks a semantic rank-1 the lexical channel missed. Run `python -m src.features.evaluation.fusion_sweep` to separate the two.

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

---

## Follow-up (2026-09-05): the reranker and Snowball arms, measured

Both remaining arms ran. Neither changes a default.

### `hybrid_word_lower_reranked` — best quality measured, unusable latency

`BAAI/bge-reranker-v2-m3` over the first 20 fused results, both channels the
baseline's own so the reranker is the only variable.

It recovers all eight questions the fusion demotes, introduces zero new misses,
passes every gate — and posts the **highest MRR of anything measured, 0.896**,
against `semantic_only`'s 0.835. It does not merely find the gold chunk; it
ranks it higher. That is what a cross-encoder is supposed to buy, and here it
demonstrably does.

The cost decides it anyway:

| | warm |
|---|---|
| p50 | 43.1 s |
| p95 | 48.5 s |
| max | 49.4 s |
| first call (cold, includes the model load) | 56.5 s |

Per query, on CPU, for 20 candidates. The deployed Space is CPU-only and nginx
and httpx both cut at 60 s, so this is not a tuning problem — it is two orders
of magnitude outside the budget. It is also not a measurement artifact:
`bge-reranker-v2-m3` is an XLM-R large (568M parameters) and this is 20 forward
passes per query. A GPU Space would change the number entirely; that is a
different deploy target and a separate decision.

Reproducibility note: two independent runs gave identical Recall@5 (0.988) and
MRR (0.896).

Measurement caveat, corrected: the first version of this latency table reported
a `max` of 158.7 s and claimed to exclude the model load. It did not —
`FlagReranker` loads its weights inside the first `rerank()` call, so sample 0
carried the load. The generator now splits the cold call out and names it, and
`test_latency_report_excludes_the_first_call_from_the_percentiles` pins that.
The percentiles above are from the corrected run.

### `hybrid_snowball_bilingual` — the first arm the gate actually rejects

Bilingual EN+ES stemming lifts Recall@5 to 0.912 from the baseline's 0.887, and
ES to 0.875. But it **loses q049 and q065**, two questions the shipped baseline
retrieves. The acceptance criterion is zero *new* misses, so it is rejected
despite the better aggregate.

This is the case the "name the questions, don't just report recall" design
exists for. Every other arm measured so far passed every gate, which made the
gates look decorative; this one shows they discriminate. An aggregate-only
report would have recorded a +0.025 improvement and hidden a swap.

### Where this leaves the lexical channel

Three independent measurements now agree that BM25 contributes nothing positive
on this corpus:

- removing it entirely (`semantic_only`) costs nothing and gains eight questions,
- repairing the fusion (`rrf_k10_sem_x2`, see `fusion_sweep_summary.md`) reaches
  the same recall and still trails on MRR,
- improving its tokenizer (Snowball) fails the zero-new-misses gate.

The cheap options are removing the channel or recalibrating the fusion; the
option that also improves ranking needs a GPU. **All three need a calibration
set separate from these 80 questions before anything is decided** — the same
discipline `REFUSAL_REVIEW_FLOOR` is held to against the Phase 3 holdout.
