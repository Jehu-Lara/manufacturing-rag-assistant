# Retrieval Evaluation Report — eval_set v1.0.0

- Embedding model: `BAAI/bge-m3`
- Fusion: Reciprocal Rank Fusion (k=60)
- Git commit: `0a56d5fc965338242400cbefcd1feea6a004be90`
- Eval set SHA-256: verified against stored hash before running

## Summary Metrics (answerable subset, n=30)

- **Recall@3**: 0.700
- **Recall@5**: 0.833
- **MRR**: 0.637

### Recall@5 by query language

- **en** (n=23): recall@5 = 0.913
- **es** (n=7): recall@5 = 0.571

## Unanswerable Subset (n=10) — Top-1 Fused Score Distribution

- min=0.0278, max=0.0328, mean=0.0318
- No refusal/gating decision is made here — Phase 3 will pick a threshold using this score distribution against the answerable subset's scores above.

## Per-Question Results (answerable subset)

| id | language | recall@3 | recall@5 | RR |
|---|---|---|---|---|
| q001 | en | 1.00 | 1.00 | 1.00 |
| q002 | en | 0.00 | 0.00 | 0.00 |
| q003 | en | 1.00 | 1.00 | 1.00 |
| q004 | en | 1.00 | 1.00 | 1.00 |
| q005 | en | 1.00 | 1.00 | 0.50 |
| q006 | en | 1.00 | 1.00 | 1.00 |
| q007 | en | 1.00 | 1.00 | 1.00 |
| q008 | en | 1.00 | 1.00 | 1.00 |
| q009 | es | 1.00 | 1.00 | 0.50 |
| q010 | es | 1.00 | 1.00 | 1.00 |
| q011 | en | 1.00 | 1.00 | 1.00 |
| q012 | es | 1.00 | 1.00 | 1.00 |
| q013 | en | 1.00 | 1.00 | 1.00 |
| q014 | es | 0.00 | 0.00 | 0.00 |
| q015 | en | 1.00 | 1.00 | 1.00 |
| q016 | es | 1.00 | 1.00 | 0.33 |
| q017 | en | 0.00 | 0.00 | 0.00 |
| q018 | es | 0.00 | 0.00 | 0.00 |
| q019 | en | 1.00 | 1.00 | 1.00 |
| q020 | en | 0.00 | 1.00 | 0.20 |
| q021 | en | 0.00 | 1.00 | 0.25 |
| q022 | en | 1.00 | 1.00 | 1.00 |
| q023 | en | 0.00 | 1.00 | 0.25 |
| q024 | en | 1.00 | 1.00 | 0.50 |
| q025 | en | 1.00 | 1.00 | 1.00 |
| q026 | es | 0.00 | 0.00 | 0.00 |
| q027 | en | 1.00 | 1.00 | 1.00 |
| q028 | en | 1.00 | 1.00 | 0.33 |
| q029 | en | 1.00 | 1.00 | 1.00 |
| q030 | en | 0.00 | 1.00 | 0.25 |

## Per-Question Results (unanswerable subset)

| id | top-1 fused score |
|---|---|
| q031 | 0.0328 |
| q032 | 0.0278 |
| q033 | 0.0315 |
| q034 | 0.0320 |
| q035 | 0.0313 |
| q036 | 0.0323 |
| q037 | 0.0328 |
| q038 | 0.0328 |
| q039 | 0.0315 |
| q040 | 0.0328 |

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

