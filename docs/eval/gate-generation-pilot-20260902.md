# Phase 3C generation pilot — 2026-09-02

## Status

**Pilot only; release gate not passed.** The write-once local run
`gate_generation_eval_20260902T225839Z` used the frozen 48-question holdout and
the five regression canaries with one holdout repeat and one canary repeat.
All seven artifact checksums were independently recomputed and matched.

This pilot did not change the shipped `binary` policy, the `0.5999` threshold,
the `0.5500` review floor, the corpus, or the `contextual-v1` index.

## Evidence bounded conclusions

- No unanswerable outcome was successfully answered in either policy arm.
  This passes the automated unsafe-answer check, but is not a general claim of
  zero hallucinations.
- Correct refusal was `1.000` for both arms. False refusal was `0.333` for
  `binary` and `0.083` for `grounded_review`: an absolute reduction of 0.25 on
  the 0–1 scale (25 percentage points; 75% relative).
- Rate limiting affected 39 outcomes and produced 48 recorded 429 events. One
  `grounded_review` canary outcome (`r002`) ended in `generation_error`.
- `r018`, `r019`, and `r020` refused in the single pilot repeat, but the
  preregistered gate requires 3/3 repeats. `r001` and `r002` did not satisfy
  their answer-and-cite gates in the pilot.
- The 38 blind-checklist rows remain ungraded, so citation accuracy,
  faithfulness, and the human safety verdict are pending.

The original report's `provider_fallbacks` value was not evidence of an actual
provider switch: the runner disables provider fallback, while the collector
historically counted `provider_call_failed` and `generation_exhausted` events
under that label. The accompanying code change records a dedicated
`provider_fallback` event and counts only that event in future runs. It does not
rewrite the sealed pilot artifacts.

The separately reported `q049` and `q072` live checks are useful smoke-test
observations, but they are not rows in this sealed holdout run and therefore are
not used as formal Phase 3C evidence here.

## Remaining gate work

Run a new single-provider evaluation with the preregistered three holdout and
three canary repeats under a quota that produces no 429s or errors. Then grade
the sealed checklist blind and import the verdicts. The default may be changed
only if every automated and human gate passes and the owner separately approves
the cost and latency trade-off.
