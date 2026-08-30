<!-- provenance
eval_set: v1.1.0, stored SHA-256 d846b3dc… (105 questions)
regression_set: v1.0.0, stored SHA-256 51874427… (20 queries)
index_profile: raw-v1  |  expansion_mode: off  |  REFUSAL_COSINE_THRESHOLD: 0.5999
retrieval: BAAI/bge-m3 + RRF k=60, existing built index under retrieval/output/
generated: Phase 1 closeout, branch fix/bilingual-refusal
method: for each answerable question, retrieve top-k, take top-5 by fused rank; classify —
  gate-over-refusal      = expected chunk IS in top-5 but 0.5999 gate refuses (retrieval OK)
  same-document-decoy    = expected chunk NOT in top-5, expected document IS in top-5, top-1 from that document
  cross-document-decoy   = expected chunk NOT in top-5, expected document IS in top-5, top-1 from a DIFFERENT document
  retrieval-miss         = expected document absent from top-5
-->

# Phase 1 failure classification — eval_set v1.1.0, raw-v1 index, expansion_mode=off

This artifact makes the design doc §5 classification reproducible from committed evidence
(supersedes the earlier hand-authored "12 decoy" estimate). Phase 2 Task 3 replaces this
with a per-question JSONL top-5 dump.

## Counts (answerable subset, n=80)

| class | count |
|---|---|
| gate-over-refusal (retrieval OK, gate refuses) | 15 |
| same-document decoy | 9 |
| cross-document decoy | 2 |
| retrieval-miss (expected document absent from top-5) | **0** |

**Non-gate retrieval failures = 11 (9 same-document + 2 cross-document). Zero true retrieval misses.**
The gate-over-refusal count (15) is larger than the decoy count (11): a gate intervention
(design §6 Phase 3) is needed in addition to contextual embedding, not instead of it.

## same-document decoys (9) — Phase 2 (C2) target

| id | lang | top-1 retrieved | expected |
|---|---|---|---|
| q002 | en | osha-3120-lockout-tagout::chunk-0002 | osha-3120-lockout-tagout::chunk-0018 |
| q014 | es | cfr-21-part-211-cgmp::chunk-0001 | cfr-21-part-211-cgmp::chunk-0024 |
| q017 | en | niosh-pocket-guide-excerpt::chunk-0006 | niosh-pocket-guide-excerpt::chunk-0010 |
| q018 | es | niosh-pocket-guide-excerpt::chunk-0007 | niosh-pocket-guide-excerpt::chunk-0018 |
| q026 | es | osha-3170-machine-guarding::chunk-0003 | osha-3170-machine-guarding::chunk-0021 |
| q066 | en | niosh-pocket-guide-excerpt::chunk-0006 | niosh-pocket-guide-excerpt::chunk-0015 |
| q067 | es | niosh-pocket-guide-excerpt::chunk-0007 | niosh-pocket-guide-excerpt::chunk-0012 |
| q075 | es | cfr-21-part-211-cgmp::chunk-0001 | cfr-21-part-211-cgmp::chunk-0014 |
| q083 | es | sop-mnt-022-cnc-mill-changeover::chunk-0002 | sop-mnt-022-cnc-mill-changeover::chunk-0005 |

Five of nine are NIOSH "How to Use This Guide / Field Definitions" front-matter chunks
(chunk-0002/0006/0007) outranking a chemical's own data chunk — the decoy pattern already
noted in the design doc §2 secondary finding.

## cross-document decoys (2)

| id | lang | top-1 retrieved | expected | note |
|---|---|---|---|---|
| q050 | en | osha-3170-machine-guarding::chunk-0021 | osha-3120-lockout-tagout::chunk-0016 | LOTO review-cycle question; machine-guarding's LOTO cross-ref chunk outranks the LOTO standard's own |
| q051 | es | osha-3170-machine-guarding::chunk-0004 | osha-3120-lockout-tagout::chunk-0012 | same shape, Spanish |

Both still place the expected *document* in the top-5, so contextual embedding may help;
they are the weakest fit for C2 and the most likely to need a ranking follow-up.
