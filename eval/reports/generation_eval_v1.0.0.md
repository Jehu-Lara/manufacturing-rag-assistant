# Generation Evaluation Report — eval_set v1.0.0

## Correct-Refusal Rate: 0.900

**False-refusal rate (answerable subset): 0.200**

Target thresholds from the plan, for reference only — this report does not compute a pass/fail verdict:
- Correct-refusal rate target: ≥ 0.80 (8/10).
- False-refusal rate target: ≤ 0.10 (≤ 3/30).

## Per-Question Results

| id | language | answerable (expected) | refused (actual) | status | confidence | threshold | retrieval_succeeded |
|---|---|---|---|---|---|---|---|
| q001 | en | True | False | ok | 0.7651 | 0.5999 | True |
| q002 | en | True | False | ok | 0.7397 | 0.5999 | False |
| q003 | en | True | False | ok | 0.7301 | 0.5999 | True |
| q004 | en | True | False | ok | 0.7154 | 0.5999 | True |
| q005 | en | True | False | ok | 0.6941 | 0.5999 | True |
| q006 | en | True | False | ok | 0.6893 | 0.5999 | True |
| q007 | en | True | False | ok | 0.7431 | 0.5999 | True |
| q008 | en | True | False | ok | 0.6361 | 0.5999 | True |
| q009 | es | True | True | ok | 0.5796 | 0.5999 | True |
| q010 | es | True | False | ok | 0.6818 | 0.5999 | True |
| q011 | en | True | False | ok | 0.6979 | 0.5999 | True |
| q012 | es | True | False | ok | 0.7111 | 0.5999 | True |
| q013 | en | True | False | ok | 0.6402 | 0.5999 | True |
| q014 | es | True | True | ok | 0.6124 | 0.5999 | False |
| q015 | en | True | False | ok | 0.7314 | 0.5999 | True |
| q016 | es | True | False | ok | 0.6134 | 0.5999 | True |
| q017 | en | True | True | ok | 0.6618 | 0.5999 | False |
| q018 | es | True | True | ok | 0.6790 | 0.5999 | False |
| q019 | en | True | False | ok | 0.8292 | 0.5999 | True |
| q020 | en | True | False | ok | 0.6723 | 0.5999 | True |
| q021 | en | True | False | ok | 0.7116 | 0.5999 | True |
| q022 | en | True | False | ok | 0.7168 | 0.5999 | True |
| q023 | en | True | False | ok | 0.6947 | 0.5999 | True |
| q024 | en | True | True | ok | 0.5837 | 0.5999 | True |
| q025 | en | True | False | ok | 0.7226 | 0.5999 | True |
| q026 | es | True | True | ok | 0.6855 | 0.5999 | False |
| q027 | en | True | False | ok | 0.6061 | 0.5999 | True |
| q028 | en | True | False | ok | 0.7441 | 0.5999 | True |
| q029 | en | True | False | ok | 0.6983 | 0.5999 | True |
| q030 | en | True | False | ok | 0.6028 | 0.5999 | True |
| q031 | en | False | True | ok | 0.5579 | 0.5999 | n/a |
| q032 | en | False | True | ok | 0.4967 | 0.5999 | n/a |
| q033 | en | False | True | ok | 0.4942 | 0.5999 | n/a |
| q034 | en | False | True | ok | 0.5968 | 0.5999 | n/a |
| q035 | en | False | True | ok | 0.5146 | 0.5999 | n/a |
| q036 | en | False | True | ok | 0.6346 | 0.5999 | n/a |
| q037 | en | False | False | ok | 0.6548 | 0.5999 | n/a |
| q038 | en | False | True | ok | 0.6261 | 0.5999 | n/a |
| q039 | en | False | True | ok | 0.4799 | 0.5999 | n/a |
| q040 | en | False | True | ok | 0.5905 | 0.5999 | n/a |

## Latency Summary (all rows, n=40)

- Mean latency: 1895.0 ms
- p90 latency: 2159.9 ms
- No per-provider breakdown: `QueryResponse` does not expose which LLM provider served a request.

## Manual Review Required

Citation accuracy and faithfulness are NOT computed by this script. Per the plan's explicit decision to grade these two headline metrics by human review rather than LLM-as-judge, see `manual_review_checklist_v1.0.0.csv` for the project owner to grade by hand (answerable subset only — unanswerable-subset correctness is already fully captured by the correct-refusal-rate metric above).

