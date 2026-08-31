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
- index_build_commit: ee928ae4c5cd27566cc5576c331f639da02783bf
- evaluation_commit: ee928ae4c5cd27566cc5576c331f639da02783bf

# Regression Eval — eval_set v1.1.0

- threshold (diagnostic): 0.5999

## expansion_mode = off

| id | lang | should_answer | top1_semantic | gate | recall@5 |
|---|---|---|---|---|---|
| r001 | en | True | 0.5642 | REFUSE | 1.0 |
| r002 | es | True | 0.5656 | REFUSE | 1.0 |
| r003 | en | True | 0.7378 | answer | 1.0 |
| r004 | es | True | 0.6818 | answer | 1.0 |
| r005 | en | True | 0.6568 | answer | 1.0 |
| r006 | en | True | 0.5125 | REFUSE | 1.0 |
| r007 | es | True | 0.5126 | REFUSE | 1.0 |
| r008 | en | True | 0.6557 | answer | 1.0 |
| r009 | es | True | 0.6081 | answer | 1.0 |
| r010 | en | True | 0.6177 | answer | 0.0 |
| r011 | es | True | 0.5813 | REFUSE | 1.0 |
| r012 | en | True | 0.6440 | answer | 1.0 |
| r013 | es | True | 0.6278 | answer | 1.0 |
| r014 | en | True | 0.6062 | answer | 1.0 |
| r015 | en | True | 0.6598 | answer | 1.0 |
| r016 | es | True | 0.6176 | answer | 1.0 |
| r017 | en | True | 0.5366 | REFUSE | 1.0 |
| r018 | en | False | 0.5630 | REFUSE | None |
| r019 | es | False | 0.5001 | REFUSE | None |
| r020 | en | False | 0.5420 | REFUSE | None |

- answerable passing gate: 11/17
- controls correctly refused: 3/3

