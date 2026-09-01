<!-- provenance-v1.1.0 -->
> **Provenance**
> - git commit: `a0c0a8cfa8d35cb3202cea30de375df488114f71`
> - commit date (report generation basis, no independent clock read): `2026-08-30T12:42:51-06:00`
> - eval_set: v1.1.0, stored SHA-256 `d846b3dc25f4bae6ba749ff5005d72e8199735c0b637c5b4fe8eb65ee54bee8c` (verified before the run)
> - regression_set: v1.0.0, stored SHA-256 `51874427db66723820fe7164845cae8dd0e7f9e59babf5f357312671bf4234e5` (verified before the run)
> - intervention config: `expansion_mode=both` (C1-both: expanded query to both channels)
> - REFUSAL_COSINE_THRESHOLD: `0.5999` (unchanged — this report measures, it does not select a threshold)
> - `eval/reports/*_v1.0.0.md` are immutable and were not regenerated or edited.

# Retrieval Evaluation Report — eval_set v1.1.0

- Embedding model: `BAAI/bge-m3`
- Fusion: Reciprocal Rank Fusion (k=60)
- Git commit: `a0c0a8cfa8d35cb3202cea30de375df488114f71`
- Eval set SHA-256: verified against stored hash before running

## Summary Metrics (answerable subset, n=80)

- **Recall@3**: 0.762
- **Recall@5**: 0.825
- **MRR**: 0.656

### Recall@5 by query language

- **en** (n=48): recall@5 = 0.917
- **es** (n=32): recall@5 = 0.688

## Matched-pair cosine gap (en − es)

Answerable en/es questions paired by set-equal `expected_chunk_ids` (ties broken by nearest id). Gap = en top-1 semantic score − es top-1 semantic score.

| en id | es id | en top-1 semantic | es top-1 semantic | gap (en − es) |
|---|---|---|---|---|
| q042 | q041 | 0.7134 | 0.7038 | +0.0096 |
| q044 | q043 | 0.7332 | 0.7165 | +0.0167 |
| q046 | q045 | 0.7250 | 0.6846 | +0.0403 |
| q048 | q047 | 0.6919 | 0.6414 | +0.0505 |
| q050 | q049 | 0.6523 | 0.6707 | -0.0184 |
| q052 | q051 | 0.7266 | 0.6130 | +0.1136 |
| q054 | q053 | 0.7114 | 0.6488 | +0.0626 |
| q056 | q055 | 0.6879 | 0.6561 | +0.0318 |
| q058 | q057 | 0.6664 | 0.6536 | +0.0128 |
| q060 | q059 | 0.6949 | 0.5929 | +0.1020 |
| q062 | q061 | 0.7799 | 0.6358 | +0.1441 |
| q064 | q063 | 0.6512 | 0.6502 | +0.0010 |
| q066 | q065 | 0.6629 | 0.6710 | -0.0081 |
| q068 | q067 | 0.7046 | 0.7404 | -0.0358 |
| q070 | q069 | 0.6046 | 0.6228 | -0.0182 |
| q072 | q071 | 0.5829 | 0.5869 | -0.0040 |
| q074 | q073 | 0.6897 | 0.6980 | -0.0083 |
| q076 | q075 | 0.6037 | 0.5718 | +0.0319 |
| q078 | q077 | 0.6527 | 0.6769 | -0.0242 |
| q080 | q079 | 0.6272 | 0.6184 | +0.0088 |
| q082 | q081 | 0.7370 | 0.6534 | +0.0835 |
| q084 | q083 | 0.6887 | 0.5964 | +0.0923 |
| q086 | q085 | 0.6146 | 0.5286 | +0.0860 |
| q088 | q087 | 0.6987 | 0.7100 | -0.0113 |
| q090 | q089 | 0.5525 | 0.5677 | -0.0152 |

- **Mean gap (en − es), n=25**: +0.0298

## Unanswerable Subset (n=25) — Top-1 Score Distribution

- Fused score: min=0.0164, max=0.0328, mean=0.0285
- Semantic score (pure cosine similarity, `semantic_rank == 1`): min=0.4271, max=0.6548, mean=0.5477
- No refusal/gating decision is made here — a separate threshold analysis picks a threshold from raw semantic_score (see `eval/reports/threshold_analysis_v1.1.0.md`), since fused_score is rank-based and disqualified as a refusal-confidence signal.

## Per-Question Results (answerable subset)

| id | language | recall@3 | recall@5 | RR | top-1 semantic score |
|---|---|---|---|---|---|
| q001 | en | 1.00 | 1.00 | 1.00 | 0.7651 |
| q002 | en | 0.00 | 0.00 | 0.00 | 0.7397 |
| q003 | en | 1.00 | 1.00 | 1.00 | 0.7301 |
| q004 | en | 1.00 | 1.00 | 1.00 | 0.7154 |
| q005 | en | 1.00 | 1.00 | 0.50 | 0.6941 |
| q006 | en | 1.00 | 1.00 | 1.00 | 0.6893 |
| q007 | en | 1.00 | 1.00 | 1.00 | 0.7431 |
| q008 | en | 1.00 | 1.00 | 1.00 | 0.6361 |
| q009 | es | 1.00 | 1.00 | 0.50 | 0.5796 |
| q010 | es | 1.00 | 1.00 | 1.00 | 0.7194 |
| q011 | en | 1.00 | 1.00 | 1.00 | 0.6979 |
| q012 | es | 1.00 | 1.00 | 1.00 | 0.7111 |
| q013 | en | 1.00 | 1.00 | 1.00 | 0.6402 |
| q014 | es | 0.00 | 0.00 | 0.00 | 0.6124 |
| q015 | en | 1.00 | 1.00 | 1.00 | 0.7314 |
| q016 | es | 1.00 | 1.00 | 0.50 | 0.6377 |
| q017 | en | 0.00 | 0.00 | 0.00 | 0.6617 |
| q018 | es | 0.00 | 0.00 | 0.00 | 0.7206 |
| q019 | en | 1.00 | 1.00 | 1.00 | 0.8292 |
| q020 | en | 0.00 | 1.00 | 0.20 | 0.6723 |
| q021 | en | 0.00 | 1.00 | 0.25 | 0.7116 |
| q022 | en | 1.00 | 1.00 | 1.00 | 0.7168 |
| q023 | en | 0.00 | 1.00 | 0.25 | 0.6947 |
| q024 | en | 1.00 | 1.00 | 0.50 | 0.5837 |
| q025 | en | 1.00 | 1.00 | 1.00 | 0.7226 |
| q026 | es | 0.00 | 0.00 | 0.00 | 0.6855 |
| q027 | en | 1.00 | 1.00 | 1.00 | 0.6061 |
| q028 | en | 1.00 | 1.00 | 0.33 | 0.7441 |
| q029 | en | 1.00 | 1.00 | 1.00 | 0.6983 |
| q030 | en | 0.00 | 1.00 | 0.25 | 0.6028 |
| q041 | es | 1.00 | 1.00 | 1.00 | 0.7038 |
| q042 | en | 1.00 | 1.00 | 1.00 | 0.7134 |
| q043 | es | 1.00 | 1.00 | 0.33 | 0.7165 |
| q044 | en | 1.00 | 1.00 | 0.50 | 0.7332 |
| q045 | es | 1.00 | 1.00 | 0.33 | 0.6846 |
| q046 | en | 1.00 | 1.00 | 1.00 | 0.7250 |
| q047 | es | 1.00 | 1.00 | 1.00 | 0.6414 |
| q048 | en | 1.00 | 1.00 | 1.00 | 0.6919 |
| q049 | es | 1.00 | 1.00 | 0.50 | 0.6707 |
| q050 | en | 1.00 | 1.00 | 0.33 | 0.6523 |
| q051 | es | 0.00 | 0.00 | 0.00 | 0.6130 |
| q052 | en | 1.00 | 1.00 | 1.00 | 0.7266 |
| q053 | es | 0.00 | 1.00 | 0.20 | 0.6488 |
| q054 | en | 1.00 | 1.00 | 0.50 | 0.7114 |
| q055 | es | 1.00 | 1.00 | 1.00 | 0.6561 |
| q056 | en | 1.00 | 1.00 | 1.00 | 0.6879 |
| q057 | es | 0.00 | 0.00 | 0.00 | 0.6536 |
| q058 | en | 1.00 | 1.00 | 0.50 | 0.6664 |
| q059 | es | 1.00 | 1.00 | 0.50 | 0.5929 |
| q060 | en | 1.00 | 1.00 | 1.00 | 0.6949 |
| q061 | es | 1.00 | 1.00 | 0.50 | 0.6358 |
| q062 | en | 1.00 | 1.00 | 1.00 | 0.7799 |
| q063 | es | 1.00 | 1.00 | 0.50 | 0.6502 |
| q064 | en | 1.00 | 1.00 | 1.00 | 0.6512 |
| q065 | es | 0.00 | 0.00 | 0.00 | 0.6710 |
| q066 | en | 0.00 | 0.00 | 0.00 | 0.6629 |
| q067 | es | 0.00 | 0.00 | 0.00 | 0.7404 |
| q068 | en | 0.00 | 0.00 | 0.00 | 0.7046 |
| q069 | es | 1.00 | 1.00 | 0.50 | 0.6228 |
| q070 | en | 1.00 | 1.00 | 1.00 | 0.6046 |
| q071 | es | 1.00 | 1.00 | 1.00 | 0.5869 |
| q072 | en | 1.00 | 1.00 | 1.00 | 0.5829 |
| q073 | es | 1.00 | 1.00 | 1.00 | 0.6980 |
| q074 | en | 1.00 | 1.00 | 1.00 | 0.6897 |
| q075 | es | 0.00 | 0.00 | 0.00 | 0.5718 |
| q076 | en | 1.00 | 1.00 | 1.00 | 0.6037 |
| q077 | es | 0.00 | 0.00 | 0.00 | 0.6769 |
| q078 | en | 1.00 | 1.00 | 1.00 | 0.6527 |
| q079 | es | 1.00 | 1.00 | 0.50 | 0.6184 |
| q080 | en | 1.00 | 1.00 | 1.00 | 0.6272 |
| q081 | es | 1.00 | 1.00 | 1.00 | 0.6534 |
| q082 | en | 1.00 | 1.00 | 1.00 | 0.7370 |
| q083 | es | 0.00 | 0.00 | 0.00 | 0.5964 |
| q084 | en | 1.00 | 1.00 | 0.50 | 0.6887 |
| q085 | es | 1.00 | 1.00 | 1.00 | 0.5286 |
| q086 | en | 1.00 | 1.00 | 1.00 | 0.6146 |
| q087 | es | 1.00 | 1.00 | 1.00 | 0.7100 |
| q088 | en | 1.00 | 1.00 | 1.00 | 0.6987 |
| q089 | es | 1.00 | 1.00 | 1.00 | 0.5677 |
| q090 | en | 1.00 | 1.00 | 1.00 | 0.5525 |

## Per-Question Results (unanswerable subset)

| id | top-1 fused score | top-1 semantic score |
|---|---|---|
| q031 | 0.0328 | 0.5579 |
| q032 | 0.0278 | 0.4967 |
| q033 | 0.0325 | 0.5598 |
| q034 | 0.0320 | 0.5968 |
| q035 | 0.0313 | 0.5146 |
| q036 | 0.0323 | 0.6346 |
| q037 | 0.0328 | 0.6548 |
| q038 | 0.0325 | 0.6356 |
| q039 | 0.0315 | 0.4799 |
| q040 | 0.0328 | 0.5905 |
| q091 | 0.0303 | 0.5509 |
| q092 | 0.0164 | 0.4271 |
| q093 | 0.0325 | 0.5882 |
| q094 | 0.0323 | 0.5519 |
| q095 | 0.0299 | 0.5065 |
| q096 | 0.0164 | 0.5816 |
| q097 | 0.0318 | 0.6286 |
| q098 | 0.0315 | 0.4504 |
| q099 | 0.0275 | 0.5518 |
| q100 | 0.0269 | 0.5407 |
| q101 | 0.0164 | 0.4867 |
| q102 | 0.0315 | 0.5246 |
| q103 | 0.0266 | 0.4687 |
| q104 | 0.0284 | 0.5565 |
| q105 | 0.0164 | 0.5582 |

## Example Queries

### q001 (en): What must an energy-control procedure include according to OSHA's lockout/tagout requirements?

Top results:
  - `osha-3120-lockout-tagout::chunk-0008` (fused=0.0320, semantic_rank=3, bm25_rank=2) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What must an energy-control procedure include?
  - `osha-3120-lockout-tagout::chunk-0007` (fused=0.0318, semantic_rank=1, bm25_rank=5) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What are OSHA's requirements?
  - `osha-3120-lockout-tagout::chunk-0016` (fused=0.0311, semantic_rank=8, bm25_rank=1) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What if a shift changes during machine service or maintenance?; How often do I need to review my lockout/tagout procedures?; What does a review entail?

### q002 (en): What is the definition of 'lockout' as used in OSHA's lockout/tagout standard?

Top results:
  - `osha-3120-lockout-tagout::chunk-0002` (fused=0.0313, semantic_rank=1, bm25_rank=7) — Control of Hazardous Energy (Lockout/Tagout) / Background > How should I use this booklet?; What is "lockout/tagout"?
  - `osha-3120-lockout-tagout::chunk-0015` (fused=0.0308, semantic_rank=7, bm25_rank=3) — Control of Hazardous Energy (Lockout/Tagout) / Requirements of the Standard > What if a group performs service or maintenance activities?
  - `osha-3170-machine-guarding::chunk-0021` (fused=0.0304, semantic_rank=2, bm25_rank=10) — Safeguarding Equipment and Protecting Employees from Amputations / Controlling Amputation Hazards > Lockout/Tagout

### q031 (en): What are the qualitative and quantitative respirator fit-testing protocols required before an employee is assigned a respirator?

Top results:
  - `niosh-pocket-guide-excerpt::chunk-0009` (fused=0.0328, semantic_rank=1, bm25_rank=1) — NIOSH Pocket Guide to Chemical Hazards — Excerpt: Common Manufacturing Chemicals / How to Use This Guide > Field Definitions
  - `niosh-pocket-guide-excerpt::chunk-0008` (fused=0.0323, semantic_rank=2, bm25_rank=2) — NIOSH Pocket Guide to Chemical Hazards — Excerpt: Common Manufacturing Chemicals / How to Use This Guide > Field Definitions
  - `osha-3151-ppe::chunk-0006` (fused=0.0302, semantic_rank=10, bm25_rank=3) — Personal Protective Equipment / Training Employees in the Proper Use of PPE

