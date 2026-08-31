<!-- provenance
source: retrieval_details_v1.1.0__contextual-v1__semantic.jsonl (per-question top-5 dump, deterministic classifier)
classifier: src/features/evaluation/failure_classification.py::classify_failure
generated: contextual-v1 / expansion_mode=semantic — deterministic classifier over retrieval_details_v1.1.0__contextual-v1__semantic.jsonl
method: for each answerable question, if it is a failure (recall@5 miss OR gate refuses a
  correctly-retrieved chunk), classify into one mutually-exclusive class:
  gate-over-refusal    = expected chunk IS in top-5 but the 0.5999 gate refuses
  same-document-decoy  = expected chunk NOT in top-5, expected document IS, top-1 from it
  cross-document-decoy = expected chunk NOT in top-5, expected document IS, top-1 from another
  retrieval-miss       = expected document absent from top-5
-->

# Failure classification — eval_set v1.1.0, contextual-v1 index, expansion_mode=semantic

Reproducible from committed machine-readable evidence: regenerate with
`python -m src.features.evaluation.failure_classification`.

## Counts (answerable subset, n=80)

| class | count |
|---|---|
| gate-over-refusal (retrieval OK, gate refuses) | 3 |
| same-document-decoy | 7 |
| cross-document-decoy | 3 |
| retrieval-miss (expected document absent from top-5) | **0** |

**Non-gate retrieval failures = 10 (7 same-document + 3 cross-document). 0 true retrieval misses.**

## Decoy failures (expected chunk absent, expected document still in top-5)

| id | lang | class | top-1 retrieved | expected |
|---|---|---|---|---|
| q002 | en | same-document-decoy | `osha-3120-lockout-tagout::chunk-0002` | `osha-3120-lockout-tagout::chunk-0018` |
| q014 | es | same-document-decoy | `cfr-21-part-211-cgmp::chunk-0001` | `cfr-21-part-211-cgmp::chunk-0024` |
| q017 | en | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0006` | `niosh-pocket-guide-excerpt::chunk-0010` |
| q026 | es | cross-document-decoy | `osha-3120-lockout-tagout::chunk-0011` | `osha-3170-machine-guarding::chunk-0021` |
| q050 | en | cross-document-decoy | `osha-3170-machine-guarding::chunk-0021` | `osha-3120-lockout-tagout::chunk-0016` |
| q051 | es | cross-document-decoy | `osha-3170-machine-guarding::chunk-0004` | `osha-3120-lockout-tagout::chunk-0012` |
| q066 | en | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0006` | `niosh-pocket-guide-excerpt::chunk-0015` |
| q067 | es | same-document-decoy | `niosh-pocket-guide-excerpt::chunk-0007` | `niosh-pocket-guide-excerpt::chunk-0012` |
| q075 | es | same-document-decoy | `cfr-21-part-211-cgmp::chunk-0001` | `cfr-21-part-211-cgmp::chunk-0014` |
| q083 | es | same-document-decoy | `sop-mnt-022-cnc-mill-changeover::chunk-0002` | `sop-mnt-022-cnc-mill-changeover::chunk-0005` |

## Gate-over-refusals (expected chunk retrieved, 0.5999 gate refuses)

| id | lang | top-1 retrieved | expected |
|---|---|---|---|
| q071 | es | `niosh-pocket-guide-excerpt::chunk-0007` | `niosh-pocket-guide-excerpt::chunk-0007` |
| q072 | en | `niosh-pocket-guide-excerpt::chunk-0002` | `niosh-pocket-guide-excerpt::chunk-0007` |
| q085 | es | `sop-qa-008-incoming-inspection::chunk-0005` | `sop-qa-008-incoming-inspection::chunk-0005` |

