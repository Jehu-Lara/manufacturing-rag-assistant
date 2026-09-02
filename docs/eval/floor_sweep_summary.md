# Review-floor sweep summary

> NOT APPLIED — mechanically eligible candidates only. Rule selection, the ADR-009 amendment, and the default flip are owner decisions gated on a fresh confirmatory holdout (v2).

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
- evaluation_commit: 8403ff28f5249c861a20a0031aa959677d631eb2
- manifest_build_commit: ee928ae4c5cd27566cc5576c331f639da02783bf

## Exhaustive candidate gates

| floor | signal | G1 | G2 | G3 | G4 | mechanically eligible |
|---|---|---|---|---|---|---|
| 0.50 | none | PASS | PASS | FAIL | FAIL | no |
| 0.50 | sem_top1_in_bm25_top_n | PASS | FAIL | FAIL | PASS | no |
| 0.50 | sem_bm25_top1_agree | PASS | FAIL | PASS | FAIL | no |
| 0.50 | channels_overlap_top_n | PASS | PASS | FAIL | FAIL | no |
| 0.51 | none | PASS | PASS | FAIL | FAIL | no |
| 0.51 | sem_top1_in_bm25_top_n | PASS | FAIL | PASS | PASS | no |
| 0.51 | sem_bm25_top1_agree | PASS | FAIL | PASS | FAIL | no |
| 0.51 | channels_overlap_top_n | PASS | PASS | FAIL | FAIL | no |
| 0.52 | none | PASS | PASS | FAIL | FAIL | no |
| 0.52 | sem_top1_in_bm25_top_n | PASS | FAIL | PASS | PASS | no |
| 0.52 | sem_bm25_top1_agree | PASS | FAIL | PASS | FAIL | no |
| 0.52 | channels_overlap_top_n | PASS | PASS | FAIL | FAIL | no |
| 0.53 | none | PASS | PASS | FAIL | FAIL | no |
| 0.53 | sem_top1_in_bm25_top_n | PASS | FAIL | PASS | PASS | no |
| 0.53 | sem_bm25_top1_agree | PASS | FAIL | PASS | FAIL | no |
| 0.53 | channels_overlap_top_n | PASS | PASS | FAIL | FAIL | no |
| 0.54 | none | PASS | PASS | PASS | FAIL | no |
| 0.54 | sem_top1_in_bm25_top_n | PASS | FAIL | PASS | PASS | no |
| 0.54 | sem_bm25_top1_agree | PASS | FAIL | PASS | FAIL | no |
| 0.54 | channels_overlap_top_n | PASS | PASS | PASS | FAIL | no |
| 0.55 (current) | none | PASS | FAIL | PASS | PASS | no |
| 0.55 | sem_top1_in_bm25_top_n | PASS | FAIL | PASS | PASS | no |
| 0.55 | sem_bm25_top1_agree | PASS | FAIL | PASS | FAIL | no |
| 0.55 | channels_overlap_top_n | PASS | FAIL | PASS | PASS | no |

## Mechanically eligible candidates

_None._

No rule was selected or applied. A fresh confirmatory holdout, paid generation evaluation, human review, and explicit owner decisions remain required.
