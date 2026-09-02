# Design — Grounded-review entry-rule calibration sweep

- **Date:** 2026-09-01
- **Status:** Implemented as an offline measurement harness; no rule selected or applied
- **Owner:** Jehu-Lara
- **Scope:** RAG4 only

## 1. Problem and evidence boundary

The frozen v1 holdout exposed two Spanish answerable questions (`h010`, `h012`)
whose expected chunks are retrieved but whose pure-semantic cosine falls below
the current `0.5500` review floor. That observation diagnoses gate
over-refusal, but it may not be used to tune a replacement rule: the holdout
has been observed and is no longer confirmatory for this decision.

This work measures candidate entry rules using only:

- `eval_set` v1.1.0 as the calibration cohort;
- `regression_queries` v1 as frozen controls; and
- the coherent local `contextual-v1/off` index.

It does not read either holdout, call an LLM, select a rule, change serving
behavior, or claim that any mechanically eligible candidate is safe to ship.

## 2. Harness

`src/features/evaluation/floor_sweep.py` retrieves every calibration/control
question exactly once with these independently named bounds:

- `RETRIEVE_K = SEMANTIC_EXTRACTION_K = 40`;
- `CHANNEL_TOP_N = DEFAULT_TOP_N = 20`; and
- `SIGNAL_TOP_N = 10`.

Per-channel ranks are therefore observable only within each channel's top 20.
Missing semantic results remain `None` and classify as `hard_refuse`, matching
serving. If a question accepts multiple expected chunks, diagnostic expected
rank is the best observed rank, never the first gold-list entry.

The immutable candidate grid is:

- floor: `(0.50, 0.51, 0.52, 0.53, 0.54, 0.55)`;
- upper threshold: the imported
  `DEFAULT_REFUSAL_COSINE_THRESHOLD` (`0.5999`), not swept; and
- signal: `none`, semantic top-1 present in BM25 top 10, semantic/BM25 top-1
  agreement, or any top-10 channel overlap.

Signals use retrieval structure only. Gold chunk ids are diagnostic fields and
never participate in candidate classification.

## 3. Candidate semantics

For each cached row:

```text
score >= 0.5999                      -> confident
score is None or score < floor       -> hard_refuse
floor <= score < 0.5999 and signal   -> grounded_review
otherwise                            -> hard_refuse
```

The current-rule cell is the imported review floor plus signal `none`.
Row-by-row equality with the real `RefusalPolicy` is a hard harness invariant.

## 4. Mechanical gates

These gates establish eligibility for later owner review, not selection:

- **G1 — fidelity:** the current cell equals
  `RefusalPolicy.classify_score()` for every cached row, including `None`.
- **G2 — false-refusal reduction:** on `eval_set`, answerable hard refusals
  decrease in at least one language and do not increase in the other.
- **G3 — containment:** relative to the current cell, at most two pooled and
  at most one per language previously hard-refused unanswerables enter review.
- **G4 — controls:** no previously admitted answerable control becomes hard
  refused; no currently hard-refused unanswerable control ascends; and the
  complete pinned set must be present with `r001/r002/r018=grounded_review`
  and `r019/r020=hard_refuse`.
- **G5 — structural preflight:** artifacts are written only beneath the
  configured calibration root, assembled in a unique `.partial` directory,
  and renamed only after all files succeed. G5 is a run-level invariant, not a
  hardcoded per-candidate boolean.

`MAX_NEWLY_REVIEWED_POOLED=2`,
`MAX_NEWLY_REVIEWED_PER_LANGUAGE=1`, and `SIGNAL_TOP_N=10` are fixed for this
calibration. Sensitivity analysis is a separate future diagnostic, not another
axis added after inspecting these results.

## 5. Outputs

Each real run produces a unique gitignored directory under
`eval/calibration/` containing:

- `features.jsonl` — per-question cached features;
- `grid.json` — complete cell statistics;
- `report.md` — exhaustive grid and G1–G4 outcomes; and
- `sanitized_summary.md` — deterministic provenance plus mechanically eligible
  candidates, without question text or gold ids.

The implementation never writes documentation. After explicit owner review,
the deterministic sanitized file may be copied byte-for-byte to
`docs/eval/floor_sweep_summary.md` as the only tracked run result.

## 6. Failure and safety behavior

- Integrity or profile mismatch aborts; the harness never rebuilds an index.
- Unknown signals fail with a client-visible `ValueError` in the offline API.
- A failed artifact write removes only its unique `.partial` directory.
- Run identifiers include UTC microseconds to avoid same-second collisions.
- No prompts, answers, credentials, provider errors, or LLM traces are read or
  written.
- The harness contains no SQL, network call, transaction, deployment, or
  serving-path mutation.

## 7. Confirmatory holdout v2 contract

Any candidate chosen later requires a new `gate_holdout_v2.0.0.json`, initially
`draft`, with at least 48 balanced EN/ES paired questions, verified chunk ids,
an atomic integrity stamp, and zero normalized-text collision with `eval_set`,
`regression_queries`, or holdout v1. Calibration and rule selection must never
read v2. No v2 file, integrity module, or questions are created in this work.

## 8. Explicit non-goals and owner gates

This work does not modify `RefusalPolicy`, config, ADR-009, corpus, retrieval
index, upper threshold, review-floor default, or deployment. It makes no paid
generation calls. Rule selection, ADR amendment, v2 authoring/freeze, paid
generation evaluation, blind human review, default flip, merge, and deployment
remain distinct owner decisions.
