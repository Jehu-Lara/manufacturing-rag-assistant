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

# Retrieval Evaluation Report — eval_set v1.1.0

- Embedding model: `BAAI/bge-m3`
- Fusion: Reciprocal Rank Fusion (k=60)
- Git commit: `ee928ae4c5cd27566cc5576c331f639da02783bf`
- Eval set SHA-256: verified against stored hash before running

## Summary Metrics (answerable subset, n=80)

- **Recall@3**: 0.825
- **Recall@5**: 0.887
- **MRR**: 0.721

### Recall@5 by query language

- **en** (n=48): recall@5 = 0.917
- **es** (n=32): recall@5 = 0.844

## Matched-pair cosine gap (en − es)

Answerable en/es questions paired by set-equal `expected_chunk_ids` (ties broken by nearest id). Gap = en top-1 semantic score − es top-1 semantic score.

| en id | es id | en top-1 semantic | es top-1 semantic | gap (en − es) |
|---|---|---|---|---|
| q042 | q041 | 0.5508 | 0.5379 | +0.0129 |
| q044 | q043 | 0.7363 | 0.7078 | +0.0285 |
| q046 | q045 | 0.7217 | 0.6833 | +0.0384 |
| q048 | q047 | 0.6947 | 0.6374 | +0.0574 |
| q050 | q049 | 0.5585 | 0.5562 | +0.0023 |
| q052 | q051 | 0.7222 | 0.6002 | +0.1220 |
| q054 | q053 | 0.7133 | 0.6187 | +0.0947 |
| q056 | q055 | 0.6825 | 0.6508 | +0.0317 |
| q058 | q057 | 0.6904 | 0.6244 | +0.0659 |
| q060 | q059 | 0.7071 | 0.6012 | +0.1059 |
| q062 | q061 | 0.7890 | 0.6511 | +0.1379 |
| q064 | q063 | 0.6628 | 0.6691 | -0.0063 |
| q066 | q065 | 0.6219 | 0.5737 | +0.0481 |
| q068 | q067 | 0.6512 | 0.6342 | +0.0169 |
| q070 | q069 | 0.6331 | 0.5924 | +0.0407 |
| q072 | q071 | 0.5414 | 0.5314 | +0.0100 |
| q074 | q073 | 0.6484 | 0.6344 | +0.0140 |
| q076 | q075 | 0.7141 | 0.6806 | +0.0335 |
| q078 | q077 | 0.6648 | 0.6879 | -0.0231 |
| q080 | q079 | 0.6592 | 0.6520 | +0.0072 |
| q082 | q081 | 0.7123 | 0.6258 | +0.0866 |
| q084 | q083 | 0.7530 | 0.6420 | +0.1111 |
| q086 | q085 | 0.6784 | 0.5853 | +0.0931 |
| q088 | q087 | 0.6887 | 0.6912 | -0.0025 |
| q090 | q089 | 0.6328 | 0.6372 | -0.0044 |

- **Mean gap (en − es), n=25**: +0.0449

## Unanswerable Subset (n=25) — Top-1 Score Distribution

- Fused score: min=0.0164, max=0.0328, mean=0.0293
- Semantic score (pure cosine similarity, `semantic_rank == 1`): min=0.4460, max=0.6575, mean=0.5433
- No refusal/gating decision is made here — a separate threshold analysis picks a threshold from raw semantic_score (see `eval/reports/threshold_analysis_v1.1.0.md`), since fused_score is rank-based and disqualified as a refusal-confidence signal.

## Per-Question Results (answerable subset)

| id | language | recall@3 | recall@5 | RR | top-1 semantic score |
|---|---|---|---|---|---|
| q001 | en | 1.00 | 1.00 | 1.00 | 0.8024 |
| q002 | en | 0.00 | 0.00 | 0.00 | 0.7296 |
| q003 | en | 1.00 | 1.00 | 1.00 | 0.7204 |
| q004 | en | 1.00 | 1.00 | 1.00 | 0.6783 |
| q005 | en | 1.00 | 1.00 | 0.50 | 0.7293 |
| q006 | en | 1.00 | 1.00 | 1.00 | 0.6988 |
| q007 | en | 1.00 | 1.00 | 1.00 | 0.7484 |
| q008 | en | 1.00 | 1.00 | 1.00 | 0.6427 |
| q009 | es | 1.00 | 1.00 | 0.50 | 0.6081 |
| q010 | es | 1.00 | 1.00 | 1.00 | 0.6818 |
| q011 | en | 1.00 | 1.00 | 1.00 | 0.7134 |
| q012 | es | 1.00 | 1.00 | 1.00 | 0.7146 |
| q013 | en | 1.00 | 1.00 | 1.00 | 0.7608 |
| q014 | es | 0.00 | 0.00 | 0.00 | 0.6977 |
| q015 | en | 1.00 | 1.00 | 1.00 | 0.7443 |
| q016 | es | 1.00 | 1.00 | 0.50 | 0.6840 |
| q017 | en | 0.00 | 0.00 | 0.00 | 0.6772 |
| q018 | es | 0.00 | 1.00 | 0.25 | 0.7036 |
| q019 | en | 1.00 | 1.00 | 1.00 | 0.8150 |
| q020 | en | 1.00 | 1.00 | 0.50 | 0.7139 |
| q021 | en | 1.00 | 1.00 | 0.33 | 0.6994 |
| q022 | en | 1.00 | 1.00 | 1.00 | 0.7368 |
| q023 | en | 1.00 | 1.00 | 1.00 | 0.6962 |
| q024 | en | 1.00 | 1.00 | 1.00 | 0.6339 |
| q025 | en | 1.00 | 1.00 | 1.00 | 0.7739 |
| q026 | es | 0.00 | 0.00 | 0.00 | 0.6876 |
| q027 | en | 1.00 | 1.00 | 1.00 | 0.6972 |
| q028 | en | 1.00 | 1.00 | 0.33 | 0.7575 |
| q029 | en | 1.00 | 1.00 | 1.00 | 0.7019 |
| q030 | en | 1.00 | 1.00 | 1.00 | 0.6813 |
| q041 | es | 1.00 | 1.00 | 1.00 | 0.5379 |
| q042 | en | 1.00 | 1.00 | 1.00 | 0.5508 |
| q043 | es | 1.00 | 1.00 | 1.00 | 0.7078 |
| q044 | en | 1.00 | 1.00 | 1.00 | 0.7363 |
| q045 | es | 1.00 | 1.00 | 0.33 | 0.6833 |
| q046 | en | 1.00 | 1.00 | 1.00 | 0.7217 |
| q047 | es | 1.00 | 1.00 | 1.00 | 0.6374 |
| q048 | en | 1.00 | 1.00 | 1.00 | 0.6947 |
| q049 | es | 1.00 | 1.00 | 0.50 | 0.5562 |
| q050 | en | 0.00 | 0.00 | 0.00 | 0.5585 |
| q051 | es | 0.00 | 0.00 | 0.00 | 0.6002 |
| q052 | en | 1.00 | 1.00 | 1.00 | 0.7222 |
| q053 | es | 0.00 | 1.00 | 0.20 | 0.6187 |
| q054 | en | 1.00 | 1.00 | 0.50 | 0.7133 |
| q055 | es | 1.00 | 1.00 | 1.00 | 0.6508 |
| q056 | en | 1.00 | 1.00 | 1.00 | 0.6825 |
| q057 | es | 0.00 | 1.00 | 0.20 | 0.6244 |
| q058 | en | 1.00 | 1.00 | 1.00 | 0.6904 |
| q059 | es | 1.00 | 1.00 | 0.50 | 0.6012 |
| q060 | en | 1.00 | 1.00 | 1.00 | 0.7071 |
| q061 | es | 1.00 | 1.00 | 0.50 | 0.6511 |
| q062 | en | 1.00 | 1.00 | 1.00 | 0.7890 |
| q063 | es | 1.00 | 1.00 | 0.50 | 0.6691 |
| q064 | en | 1.00 | 1.00 | 1.00 | 0.6628 |
| q065 | es | 0.00 | 1.00 | 0.25 | 0.5737 |
| q066 | en | 0.00 | 0.00 | 0.00 | 0.6219 |
| q067 | es | 1.00 | 1.00 | 0.33 | 0.6342 |
| q068 | en | 1.00 | 1.00 | 0.33 | 0.6512 |
| q069 | es | 1.00 | 1.00 | 1.00 | 0.5924 |
| q070 | en | 1.00 | 1.00 | 1.00 | 0.6331 |
| q071 | es | 1.00 | 1.00 | 0.50 | 0.5314 |
| q072 | en | 0.00 | 1.00 | 0.25 | 0.5414 |
| q073 | es | 1.00 | 1.00 | 1.00 | 0.6344 |
| q074 | en | 1.00 | 1.00 | 1.00 | 0.6484 |
| q075 | es | 0.00 | 0.00 | 0.00 | 0.6806 |
| q076 | en | 1.00 | 1.00 | 1.00 | 0.7141 |
| q077 | es | 1.00 | 1.00 | 0.33 | 0.6879 |
| q078 | en | 1.00 | 1.00 | 1.00 | 0.6648 |
| q079 | es | 1.00 | 1.00 | 0.50 | 0.6520 |
| q080 | en | 1.00 | 1.00 | 1.00 | 0.6592 |
| q081 | es | 1.00 | 1.00 | 1.00 | 0.6258 |
| q082 | en | 1.00 | 1.00 | 1.00 | 0.7123 |
| q083 | es | 0.00 | 0.00 | 0.00 | 0.6420 |
| q084 | en | 1.00 | 1.00 | 1.00 | 0.7530 |
| q085 | es | 1.00 | 1.00 | 1.00 | 0.5853 |
| q086 | en | 1.00 | 1.00 | 1.00 | 0.6784 |
| q087 | es | 1.00 | 1.00 | 1.00 | 0.6912 |
| q088 | en | 1.00 | 1.00 | 1.00 | 0.6887 |
| q089 | es | 1.00 | 1.00 | 1.00 | 0.6372 |
| q090 | en | 1.00 | 1.00 | 1.00 | 0.6328 |

## Per-Question Results (unanswerable subset)

| id | top-1 fused score | top-1 semantic score |
|---|---|---|
| q031 | 0.0328 | 0.5306 |
| q032 | 0.0285 | 0.5007 |
| q033 | 0.0318 | 0.4817 |
| q034 | 0.0320 | 0.5936 |
| q035 | 0.0313 | 0.5038 |
| q036 | 0.0323 | 0.6278 |
| q037 | 0.0328 | 0.6575 |
| q038 | 0.0328 | 0.6355 |
| q039 | 0.0305 | 0.4788 |
| q040 | 0.0328 | 0.5931 |
| q091 | 0.0311 | 0.5578 |
| q092 | 0.0260 | 0.4460 |
| q093 | 0.0315 | 0.5497 |
| q094 | 0.0317 | 0.5525 |
| q095 | 0.0284 | 0.4938 |
| q096 | 0.0267 | 0.5841 |
| q097 | 0.0318 | 0.6314 |
| q098 | 0.0305 | 0.4604 |
| q099 | 0.0271 | 0.5513 |
| q100 | 0.0289 | 0.5244 |
| q101 | 0.0164 | 0.4908 |
| q102 | 0.0317 | 0.5367 |
| q103 | 0.0276 | 0.4755 |
| q104 | 0.0284 | 0.5513 |
| q105 | 0.0164 | 0.5746 |

## Example Queries

### q001 (en): What must an energy-control procedure include according to OSHA's lockout/tagout requirements?

Top results:
  - `osha-3120-lockout-tagout::chunk-0008` (fused=0.0323, semantic_rank=2, bm25_rank=2) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What must an energy-control procedure include?
  - `osha-3120-lockout-tagout::chunk-0007` (fused=0.0318, semantic_rank=1, bm25_rank=5) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What are OSHA's requirements?
  - `osha-3120-lockout-tagout::chunk-0016` (fused=0.0311, semantic_rank=8, bm25_rank=1) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What if a shift changes during machine service or maintenance?; How often do I need to review my lockout/tagout procedures?; What does a review entail?

### q002 (en): What is the definition of 'lockout' as used in OSHA's lockout/tagout standard?

Top results:
  - `osha-3120-lockout-tagout::chunk-0002` (fused=0.0313, semantic_rank=1, bm25_rank=7) — Control of Hazardous Energy (Lockout/Tagout) / Background > How should I use this booklet?; What is "lockout/tagout"?
  - `osha-3120-lockout-tagout::chunk-0015` (fused=0.0302, semantic_rank=10, bm25_rank=3) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What if a group performs service or maintenance activities?
  - `osha-3170-machine-guarding::chunk-0021` (fused=0.0302, semantic_rank=3, bm25_rank=10) — Safeguarding Equipment and Protecting Employees from Amputations / Controlling Amputation Hazards > Lockout/Tagout

### q031 (en): What are the qualitative and quantitative respirator fit-testing protocols required before an employee is assigned a respirator?

Top results:
  - `niosh-pocket-guide-excerpt::chunk-0009` (fused=0.0328, semantic_rank=1, bm25_rank=1) — NIOSH Pocket Guide to Chemical Hazards — Excerpt: Common Manufacturing Chemicals / How to Use This Guide > Field Definitions
  - `niosh-pocket-guide-excerpt::chunk-0008` (fused=0.0313, semantic_rank=6, bm25_rank=2) — NIOSH Pocket Guide to Chemical Hazards — Excerpt: Common Manufacturing Chemicals / How to Use This Guide > Field Definitions
  - `cfr-21-part-211-cgmp::chunk-0005` (fused=0.0301, semantic_rank=9, bm25_rank=4) — 21 CFR Part 211 — Current Good Manufacturing Practice for Finished Pharmaceuticals / Subpart B—Organization and Personnel > § 211.25 Personnel qualifications

