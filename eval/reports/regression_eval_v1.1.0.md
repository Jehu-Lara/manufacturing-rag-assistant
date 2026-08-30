<!-- provenance-v1.1.0 -->
> **Provenance**
> - git commit: `a0c0a8cfa8d35cb3202cea30de375df488114f71`
> - commit date (report generation basis, no independent clock read): `2026-08-30T12:42:51-06:00`
> - eval_set: v1.1.0, stored SHA-256 `d846b3dc25f4bae6ba749ff5005d72e8199735c0b637c5b4fe8eb65ee54bee8c` (verified before the run)
> - regression_set: v1.0.0, stored SHA-256 `51874427db66723820fe7164845cae8dd0e7f9e59babf5f357312671bf4234e5` (verified before the run)
> - intervention config: all four configs (`off`, `semantic`, `lexical`, `both`), one section each
> - REFUSAL_COSINE_THRESHOLD: `0.5999` (unchanged — this report measures, it does not select a threshold)
> - `eval/reports/*_v1.0.0.md` are immutable and were not regenerated or edited.

# Regression Eval — eval_set v1.1.0

- threshold (diagnostic): 0.5999

## expansion_mode = off

| id | lang | should_answer | top1_semantic | gate | recall@5 |
|---|---|---|---|---|---|
| r001 | en | True | 0.5582 | REFUSE | 1.0 |
| r002 | es | True | 0.5598 | REFUSE | 1.0 |
| r003 | en | True | 0.7295 | answer | 1.0 |
| r004 | es | True | 0.6818 | answer | 1.0 |
| r005 | en | True | 0.6468 | answer | 1.0 |
| r006 | en | True | 0.4988 | REFUSE | 1.0 |
| r007 | es | True | 0.5011 | REFUSE | 1.0 |
| r008 | en | True | 0.6461 | answer | 1.0 |
| r009 | es | True | 0.5796 | REFUSE | 1.0 |
| r010 | en | True | 0.6131 | answer | 0.0 |
| r011 | es | True | 0.5303 | REFUSE | 1.0 |
| r012 | en | True | 0.6357 | answer | 1.0 |
| r013 | es | True | 0.6262 | answer | 1.0 |
| r014 | en | True | 0.5876 | REFUSE | 1.0 |
| r015 | en | True | 0.6730 | answer | 1.0 |
| r016 | es | True | 0.6280 | answer | 1.0 |
| r017 | en | True | 0.5285 | REFUSE | 1.0 |
| r018 | en | False | 0.5623 | REFUSE | None |
| r019 | es | False | 0.5018 | REFUSE | None |
| r020 | en | False | 0.4963 | REFUSE | None |

- answerable passing gate: 9/17
- controls correctly refused: 3/3

## expansion_mode = semantic

| id | lang | should_answer | top1_semantic | gate | recall@5 |
|---|---|---|---|---|---|
| r001 | en | True | 0.7195 | answer | 1.0 |
| r002 | es | True | 0.7193 | answer | 1.0 |
| r003 | en | True | 0.7236 | answer | 1.0 |
| r004 | es | True | 0.7194 | answer | 1.0 |
| r005 | en | True | 0.6468 | answer | 1.0 |
| r006 | en | True | 0.6940 | answer | 1.0 |
| r007 | es | True | 0.6897 | answer | 1.0 |
| r008 | en | True | 0.6461 | answer | 1.0 |
| r009 | es | True | 0.5796 | REFUSE | 1.0 |
| r010 | en | True | 0.6333 | answer | 0.0 |
| r011 | es | True | 0.5987 | REFUSE | 1.0 |
| r012 | en | True | 0.6819 | answer | 1.0 |
| r013 | es | True | 0.7183 | answer | 1.0 |
| r014 | en | True | 0.5876 | REFUSE | 1.0 |
| r015 | en | True | 0.7206 | answer | 1.0 |
| r016 | es | True | 0.6873 | answer | 1.0 |
| r017 | en | True | 0.6607 | answer | 1.0 |
| r018 | en | False | 0.6719 | answer | None |
| r019 | es | False | 0.6638 | answer | None |
| r020 | en | False | 0.5892 | REFUSE | None |

- answerable passing gate: 14/17
- controls correctly refused: 1/3

## expansion_mode = lexical

| id | lang | should_answer | top1_semantic | gate | recall@5 |
|---|---|---|---|---|---|
| r001 | en | True | 0.5582 | REFUSE | 1.0 |
| r002 | es | True | 0.5598 | REFUSE | 1.0 |
| r003 | en | True | 0.7295 | answer | 1.0 |
| r004 | es | True | 0.6818 | answer | 1.0 |
| r005 | en | True | 0.6468 | answer | 1.0 |
| r006 | en | True | 0.4988 | REFUSE | 1.0 |
| r007 | es | True | 0.5011 | REFUSE | 1.0 |
| r008 | en | True | 0.6461 | answer | 1.0 |
| r009 | es | True | 0.5796 | REFUSE | 1.0 |
| r010 | en | True | 0.6131 | answer | 0.0 |
| r011 | es | True | 0.5303 | REFUSE | 1.0 |
| r012 | en | True | 0.6357 | answer | 0.0 |
| r013 | es | True | 0.6262 | answer | 0.0 |
| r014 | en | True | 0.5876 | REFUSE | 1.0 |
| r015 | en | True | 0.6730 | answer | 1.0 |
| r016 | es | True | 0.6280 | answer | 1.0 |
| r017 | en | True | 0.5285 | REFUSE | 1.0 |
| r018 | en | False | 0.5623 | REFUSE | None |
| r019 | es | False | 0.5018 | REFUSE | None |
| r020 | en | False | 0.4963 | REFUSE | None |

- answerable passing gate: 9/17
- controls correctly refused: 3/3

## expansion_mode = both

| id | lang | should_answer | top1_semantic | gate | recall@5 |
|---|---|---|---|---|---|
| r001 | en | True | 0.7195 | answer | 1.0 |
| r002 | es | True | 0.7193 | answer | 1.0 |
| r003 | en | True | 0.7236 | answer | 1.0 |
| r004 | es | True | 0.7194 | answer | 1.0 |
| r005 | en | True | 0.6468 | answer | 1.0 |
| r006 | en | True | 0.6940 | answer | 1.0 |
| r007 | es | True | 0.6897 | answer | 1.0 |
| r008 | en | True | 0.6461 | answer | 1.0 |
| r009 | es | True | 0.5796 | REFUSE | 1.0 |
| r010 | en | True | 0.6333 | answer | 0.0 |
| r011 | es | True | 0.5987 | REFUSE | 1.0 |
| r012 | en | True | 0.6819 | answer | 1.0 |
| r013 | es | True | 0.7183 | answer | 0.0 |
| r014 | en | True | 0.5876 | REFUSE | 1.0 |
| r015 | en | True | 0.7206 | answer | 1.0 |
| r016 | es | True | 0.6873 | answer | 1.0 |
| r017 | en | True | 0.6607 | answer | 1.0 |
| r018 | en | False | 0.6719 | answer | None |
| r019 | es | False | 0.6638 | answer | None |
| r020 | en | False | 0.5892 | REFUSE | None |

- answerable passing gate: 14/17
- controls correctly refused: 1/3

