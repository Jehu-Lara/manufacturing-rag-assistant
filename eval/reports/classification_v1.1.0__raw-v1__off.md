<!-- provenance
source: retrieval_details_v1.1.0__raw-v1__off.jsonl (per-question top-5 dump, deterministic classifier)
classifier: src/features/evaluation/failure_classification.py::classify_failure
generated: raw-v1 / expansion_mode=off — deterministic classifier over retrieval_details_v1.1.0__raw-v1__off.jsonl
method: for each answerable question, if it is a failure (recall@5 miss OR gate refuses a
  correctly-retrieved chunk), classify into one mutually-exclusive class:
  gate-over-refusal    = expected chunk IS in top-5 but the 0.5999 gate refuses
  same-document-decoy  = expected chunk NOT in top-5, expected document IS, top-1 from it
  cross-document-decoy = expected chunk NOT in top-5, expected document IS, top-1 from another
  retrieval-miss       = expected document absent from top-5
-->

# Failure classification — eval_set v1.1.0, raw-v1 index, expansion_mode=off

Reproducible from committed machine-readable evidence: regenerate with
`python -m src.features.evaluation.failure_classification`.

## Counts (answerable subset, n=80)

| class | count |
|---|---|
| gate-over-refusal (retrieval OK, gate refuses) | 15 |
| same-document-decoy | 9 |
| cross-document-decoy | 2 |
| retrieval-miss (expected document absent from top-5) | **0** |

**Non-gate retrieval failures = 11 (9 same-document + 2 cross-document). 0 true retrieval misses.**

## Decoy failures (expected chunk absent, expected document still in top-5)

| id | lang | class | top-1 retrieved | expected |
|---|---|---|---|---|
| q002 | en | same-document-decoy | `osha-3120-lockout-tagout::chunk-0002` | `osha-3120-lockout-tagout::chunk-0018` |
| q014 | es | same-document-decoy | `cfr-21-part-211-cgmp::chunk-0001` | `cfr-21-part-211-cgmp::chunk-0024` |
| q017 | en | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0006` | `niosh-pocket-guide-excerpt::chunk-0010` |
| q018 | es | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0007` | `niosh-pocket-guide-excerpt::chunk-0018` |
| q026 | es | same-document-decoy | `osha-3170-machine-guarding::chunk-0003` | `osha-3170-machine-guarding::chunk-0021` |
| q050 | en | cross-document-decoy | `osha-3170-machine-guarding::chunk-0021` | `osha-3120-lockout-tagout::chunk-0016` |
| q051 | es | cross-document-decoy | `osha-3170-machine-guarding::chunk-0004` | `osha-3120-lockout-tagout::chunk-0012` |
| q066 | en | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0006` | `niosh-pocket-guide-excerpt::chunk-0015` |
| q067 | es | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0007` | `niosh-pocket-guide-excerpt::chunk-0012` |
| q075 | es | same-document-decoy | `cfr-21-part-211-cgmp::chunk-0001` | `cfr-21-part-211-cgmp::chunk-0014` |
| q083 | es | same-document-decoy | `sop-mnt-022-cnc-mill-changeover::chunk-0002` | `sop-mnt-022-cnc-mill-changeover::chunk-0005` |

## Gate-over-refusals (expected chunk retrieved, 0.5999 gate refuses)

| id | lang | top-1 retrieved | expected |
|---|---|---|---|
| q009 | es | `doe-hdbk-1018-2-valves::chunk-0006` | `doe-hdbk-1018-1-pumps::chunk-0006` |
| q024 | en | `manual-xj450-belt-conveyor::chunk-0006` | `cmms-work-orders-line3-q2::chunk-0003` |
| q041 | es | `doe-hdbk-1018-1-pumps::chunk-0008` | `doe-hdbk-1018-1-pumps::chunk-0007` |
| q042 | en | `doe-hdbk-1018-1-pumps::chunk-0007` | `doe-hdbk-1018-1-pumps::chunk-0007` |
| q049 | es | `osha-3170-machine-guarding::chunk-0021` | `osha-3120-lockout-tagout::chunk-0016` |
| q059 | es | `doe-hdbk-1018-2-valves::chunk-0016` | `osha-3151-ppe::chunk-0015` |
| q065 | es | `niosh-pocket-guide-excerpt::chunk-0006` | `niosh-pocket-guide-excerpt::chunk-0015` |
| q069 | es | `niosh-pocket-guide-excerpt::chunk-0011` | `niosh-pocket-guide-excerpt::chunk-0011` |
| q070 | en | `niosh-pocket-guide-excerpt::chunk-0011` | `niosh-pocket-guide-excerpt::chunk-0011` |
| q071 | es | `niosh-pocket-guide-excerpt::chunk-0016` | `niosh-pocket-guide-excerpt::chunk-0007` |
| q072 | en | `niosh-pocket-guide-excerpt::chunk-0002` | `niosh-pocket-guide-excerpt::chunk-0007` |
| q073 | es | `cfr-21-part-211-cgmp::chunk-0005` | `cfr-21-part-211-cgmp::chunk-0005` |
| q085 | es | `sop-qa-008-incoming-inspection::chunk-0005` | `sop-qa-008-incoming-inspection::chunk-0005` |
| q089 | es | `cmms-work-orders-line3-q2::chunk-0006` | `cmms-work-orders-line3-q2::chunk-0006` |
| q090 | en | `cmms-work-orders-line3-q2::chunk-0006` | `cmms-work-orders-line3-q2::chunk-0006` |

