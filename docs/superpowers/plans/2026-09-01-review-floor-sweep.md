# Grounded-review entry-rule calibration sweep — Implementation plan

**Goal:** Produce an offline, reproducible calibration grid without selecting
or applying a serving rule.

**Base:** `master@e0800e0`; PR #5 remains independent.

**Spec:** `docs/superpowers/specs/2026-09-01-review-floor-sweep-design.md`

## Fixed decisions

- `SIGNAL_TOP_N=10` is fixed, not swept.
- Containment caps are two pooled and one per language.
- Results are exhaustive/eligible, not ranked or selected.
- Generated runs are ignored; the tracked summary is a reviewed byte-for-byte
  copy of the deterministic sanitized artifact.
- Implementation, real sweep, commit, push, PR, paid evaluation, merge, and
  deployment are separate gates. This plan contains no automatic merge or
  deployment.

## Task 1 — Fail-first feature and classifier tests

Files:

- create `tests/test_evaluation_floor_sweep.py`;
- create `src/features/evaluation/floor_sweep.py`; and
- add `eval/calibration/` to `.gitignore`.

Tests must establish:

- constants are immutable and imported from existing config/retrieval values;
- multiple expected chunks use the best rank;
- empty semantic results remain representable and hard-refuse;
- the current candidate equals real `RefusalPolicy` for boundary values;
- unknown signals raise; and
- channel signals affect only the review band, never confident classification.

Acceptance:

```powershell
python -m pytest tests/test_evaluation_floor_sweep.py -q --basetemp=.pytest_tmp_floor
ruff check src/features/evaluation/floor_sweep.py tests/test_evaluation_floor_sweep.py
mypy src/features/evaluation/floor_sweep.py
```

## Task 2 — Grid and gates

Implement immutable `FeatureRow`, `CellStat`, and `RuleAssessment` records;
`classify_candidate`; `sweep_grid`; `assess_rule`; `assess_grid`; and
`eligible_rules`.

Tests must prove:

- every cell partitions its cohort/language/class slice;
- G3 rejects pooled and per-language over-promotion;
- G4 fails when even one pinned control is missing;
- G4 accepts the complete current baseline; and
- G1 compares the same cached row against real policy exactly.

G5 must not appear in `RuleAssessment`: it is a run-level artifact invariant.

## Task 3 — Pure artifact writer

Implement `write_run(rows, manifest, calibration_dir, now=None)` separately
from real retrieval. Unit tests inject `tmp_path`, synthetic rows, a fake
manifest timestamp, and no model/index.

Required behavior:

1. Resolve the injected calibration root.
2. Create a microsecond-unique `.partial` directory beneath it.
3. Write LF-stable `features.jsonl`, `grid.json`, `report.md`, and
   `sanitized_summary.md`.
4. Rename to the final directory only after all writes succeed.
5. On failure, remove only that exact `.partial` directory and re-raise.

No test may infer confinement from `git status`: ignored files make that check
vacuous. Assert paths and filesystem contents directly.

## Task 4 — Real orchestration and contamination guard

`collect_rows()` must verify both input datasets and the existing
`contextual-v1` manifest/physical indexes, then retrieve each of the 105 eval
questions and 20 regression controls once using `off`, `k=40`, `top_n=20`.
It must abort rather than rebuild an absent or mismatched index.

Add an AST guard proving serving code and `floor_sweep.py` do not import or
name holdout v1/v2. Existing holdout-specific evaluation modules remain out of
the scan by construction.

## Task 5 — Verification before real measurement

Run with a unique external pytest base directory. Git-sensitive tests require
`tmp_path` to live outside the repository so Git cannot discover the parent
checkout. Do not reuse pytest's shared Windows temp root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$testTemp = Join-Path ([IO.Path]::GetTempPath()) ('rag4-pytest-' + [guid]::NewGuid())
python -m pytest tests/test_evaluation_floor_sweep.py -q --basetemp=$testTemp -p no:cacheprovider
ruff check src tests
mypy src
python -m pytest -q --basetemp=$testTemp -p no:cacheprovider
```

Delete only the verified unique external temporary directory after the
processes finish. Any failing gate stops the work; do not weaken thresholds or
tests to obtain green output.

## Task 6 — Owner-authorized free sweep and tracked summary

With `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, run:

```powershell
python -m src.features.evaluation.floor_sweep
```

Verify the finalized directory has exactly four files and no `.partial`
sibling. Review `sanitized_summary.md`, then copy it byte-for-byte to
`docs/eval/floor_sweep_summary.md`. Do not copy questions, feature rows, or
gold ids into tracked documentation.

## Task 7 — Delivery

After a clean diff review, create commits scoped to implementation/docs,
push `codex/review-floor-sweep`, and open PR #6 against `master`. The PR body
must state:

- no rule was selected or applied;
- no holdout was read by the harness;
- no LLM/provider call occurred;
- PR #5 remains independent;
- local validation commands and exact results; and
- next gates: choose whether to pursue an eligible rule, create/freeze a fresh
  v2 holdout, run paid causal generation, perform blind human review, and only
  then consider config/ADR/default/deployment changes.

Do not merge or deploy from this plan.
