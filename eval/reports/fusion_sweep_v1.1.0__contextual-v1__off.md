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
- evaluation_commit: 972558f390f045b5c24d852291134e65a4422f3c

# RRF fusion parameter sweep

Evaluation set v1.1.0, live index `contextual-v1`, `expansion_mode=off`, Recall@5 over the answerable subset. Channel outputs captured once per question and re-fused offline, which is exact — fusion is pure.

**Nothing here changes a default.** RRF `k=60` and the ascending-`chunk_id` tie-break are byte-stable invariants; this run reads them and writes nothing. Every row is a measurement, not a decision.

| rule | Recall@5 | EN | ES | MRR | new misses | rescues |
|---|---|---|---|---|---|---|
| `rrf_k60` | 0.887 | 0.917 | 0.844 | 0.721 | 0 | 0 |
| `rrf_k20` | 0.900 | 0.917 | 0.875 | 0.725 | 0 | 1 |
| `rrf_k10` | 0.938 | 0.938 | 0.938 | 0.740 | 0 | 4 |
| `rrf_k5` | 0.950 | 0.958 | 0.938 | 0.765 | 0 | 5 |
| `rrf_k1` | 0.975 | 0.979 | 0.969 | 0.769 | 0 | 7 |
| `rrf_k60_sem_x2` | 0.900 | 0.917 | 0.875 | 0.731 | 0 | 1 |
| `rrf_k60_sem_x3` | 0.900 | 0.917 | 0.875 | 0.727 | 0 | 1 |
| `rrf_k10_sem_x2` | 0.988 | 0.979 | 1.000 | 0.798 | 0 | 8 |
| `semantic_only` | 0.988 | 0.979 | 1.000 | 0.835 | 0 | 8 |

## Acceptance gates

- `rrf_k60` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k20` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k10` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k5` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k1` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k60_sem_x2` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k60_sem_x3` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `rrf_k10_sem_x2` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS
- `semantic_only` — EN Recall@5 >= 0.917: PASS; ES Recall@5 >= 0.844: PASS; Global Recall@5 >= 0.887: PASS; zero new misses: PASS

## Per-question deltas vs the shipped `rrf_k60`

### `rrf_k20`

- Recall@5 delta: +0.0125
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q026']

### `rrf_k10`

- Recall@5 delta: +0.0500
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q014', 'q026', 'q050', 'q075']

### `rrf_k5`

- Recall@5 delta: +0.0625
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q014', 'q026', 'q050', 'q066', 'q075']

### `rrf_k1`

- Recall@5 delta: +0.0875
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075']

### `rrf_k60_sem_x2`

- Recall@5 delta: +0.0125
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q026']

### `rrf_k60_sem_x3`

- Recall@5 delta: +0.0125
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q026']

### `rrf_k10_sem_x2`

- Recall@5 delta: +0.1000
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083']

### `semantic_only`

- Recall@5 delta: +0.1000
- New misses (shipped found, rule lost): none
- Rescues (shipped lost, rule found): ['q014', 'q017', 'q026', 'q050', 'q051', 'q066', 'q075', 'q083']

## Where the gold chunk actually sat

For every question the shipped fusion loses and the semantic channel alone retrieves, the gold chunk's rank in each raw channel before fusion:

| question | semantic rank | BM25 rank |
|---|---|---|
| `q014` | 1 | — |
| `q017` | 2 | — |
| `q026` | 1 | 19 |
| `q050` | 1 | — |
| `q051` | 1 | — |
| `q066` | 2 | — |
| `q075` | 1 | — |
| `q083` | 3 | — |

A gold chunk sitting at semantic rank 1-3 and landing outside the fused top 5 is a fusion result, not a retrieval failure.

