## Provenance
- eval_set_version: 1.1.0
- eval_set_sha256: d846b3dc25f4bae6ba749ff5005d72e8199735c0b637c5b4fe8eb65ee54bee8c
- regression_set_version: 1.0.0
- regression_set_sha256: 51874427db66723820fe7164845cae8dd0e7f9e59babf5f357312671bf4234e5
- index_profile: contextual-v1
- expansion_mode: off
- refusal_cosine_threshold: 0.5999
- chunks_sha256: dd7d21b5bc888a955e62a41c7fd1c0809739cfb8c4718137c4f89420b3fa17bf
- corpus_sha256: 2fc786caa428b56d365ab85d66bfcdc096dab0728db70996a2982d72463afa21
- embedding_model: BAAI/bge-m3
- embedding_revision: 5617a9f61b028005a4858fdac845db406aefb181
- index_build_commit: 2e7bd069a18873c03cf7a5144bc612e4675393e1
- evaluation_commit: e6f3e090225bbb25adc32a6ec26dab3fe5c12caa

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

