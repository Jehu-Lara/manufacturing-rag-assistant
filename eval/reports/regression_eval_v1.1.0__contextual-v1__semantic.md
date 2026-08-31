## Provenance
- eval_set_version: 1.1.0
- eval_set_sha256: d846b3dc25f4bae6ba749ff5005d72e8199735c0b637c5b4fe8eb65ee54bee8c
- regression_set_version: 1.0.0
- regression_set_sha256: 51874427db66723820fe7164845cae8dd0e7f9e59babf5f357312671bf4234e5
- index_profile: contextual-v1
- expansion_mode: semantic
- refusal_cosine_threshold: 0.5999
- chunks_sha256: dd7d21b5bc888a955e62a41c7fd1c0809739cfb8c4718137c4f89420b3fa17bf
- corpus_sha256: 2fc786caa428b56d365ab85d66bfcdc096dab0728db70996a2982d72463afa21
- embedding_model: BAAI/bge-m3
- embedding_revision: 5617a9f61b028005a4858fdac845db406aefb181
- index_build_commit: ee928ae4c5cd27566cc5576c331f639da02783bf
- evaluation_commit: ee928ae4c5cd27566cc5576c331f639da02783bf

# Regression Eval — eval_set v1.1.0

- threshold (diagnostic): 0.5999

## expansion_mode = semantic

| id | lang | should_answer | top1_semantic | gate | recall@5 |
|---|---|---|---|---|---|
| r001 | en | True | 0.7128 | answer | 1.0 |
| r002 | es | True | 0.7124 | answer | 1.0 |
| r003 | en | True | 0.7223 | answer | 1.0 |
| r004 | es | True | 0.7144 | answer | 1.0 |
| r005 | en | True | 0.6568 | answer | 1.0 |
| r006 | en | True | 0.6944 | answer | 1.0 |
| r007 | es | True | 0.6895 | answer | 1.0 |
| r008 | en | True | 0.6557 | answer | 1.0 |
| r009 | es | True | 0.6081 | answer | 1.0 |
| r010 | en | True | 0.6486 | answer | 0.0 |
| r011 | es | True | 0.6136 | answer | 1.0 |
| r012 | en | True | 0.7028 | answer | 1.0 |
| r013 | es | True | 0.7287 | answer | 1.0 |
| r014 | en | True | 0.6062 | answer | 1.0 |
| r015 | en | True | 0.7170 | answer | 1.0 |
| r016 | es | True | 0.6848 | answer | 1.0 |
| r017 | en | True | 0.6611 | answer | 1.0 |
| r018 | en | False | 0.6736 | answer | None |
| r019 | es | False | 0.6660 | answer | None |
| r020 | en | False | 0.6004 | answer | None |

- answerable passing gate: 17/17
- controls correctly refused: 0/3

