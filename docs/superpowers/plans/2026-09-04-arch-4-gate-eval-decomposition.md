# Bucket 4 — `gate_generation_eval.py` Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1143-line `src/features/evaluation/gate_generation_eval.py` into four focused modules — models, runner, artifacts, verdicts — behind a compatibility façade that keeps `python -m src.features.evaluation.gate_generation_eval` and every existing import working byte-for-byte. `src.features.evaluation` stays where it is; there is no mass move to `src/eval`.

**Architecture:** Four ordered tasks, each a pure move plus re-export, with a golden-artifact test standing guard the whole way. T0 (Task 1) captures the current run's artifact bytes as a fixture so any later drift is caught immediately. T2 extracts the leaf-most layer (models), T3 the artifact writers, T4 the verdict/gate layer, leaving the façade holding `run`, `main`, and the re-exports. Nothing about the run's semantics, sealing, checksums or gate arithmetic changes.

**Tech Stack:** Python 3.11, pytest, dataclasses, csv/json/hashlib (stdlib).

**Spec:** `docs/superpowers/specs/2026-09-04-architecture-remediation-design.md`

## Global Constraints

- Execution is gated: PR #9 resolved by the owner, then a new branch cut from `master` with the owner's authorization.
- **No paid run.** Every test in this bucket uses the injected fakes `tests/test_evaluation_gate_generation_eval.py` already provides. `gate_generation_eval` is not wired into CI and must not be.
- **The artifact contract is frozen.** `run_manifest.json`, `retrieval.jsonl`, `outcomes.jsonl`, `blind_checklist.csv`, `blind_checklist.baseline.json`, `arm_map.sealed.json`, `checksums.txt`, `comparison.md` keep identical filenames, identical column order, identical JSON key order and identical checksum semantics. An existing run directory must still re-verify after this refactor.
- Write-once semantics stay: build in `<id>.partial/`, rename to `<id>/`; imports are single-use and fail closed on an existing `checksums.import.txt`; sealed files are never rewritten by `--import-verdicts`.
- Gate arithmetic is untouched — `repeats_conform`, `no_rate_limiting`, canary must-answer/must-refuse, grey-band coverage, the `FULL_REPEATS = 3` / `CANARY_REPEATS = 3` matrix. This is a move, not a redesign.
- Byte-stable invariants unchanged (`PINNED_THRESHOLD = 0.5999`, `PINNED_REVIEW_FLOOR = 0.5500`, RRF `k=60`, `binary`, `off`, `contextual-v1`).
- After each task: `pytest tests/test_evaluation_gate_generation_eval.py -q` green. End of bucket: `pytest -q`, `ruff check src tests`, `mypy src` green.

---

## File Structure

New package `src/features/evaluation/gate_eval/`:

- `models.py` — `TraceCollector` (61), `WithinRepeatCache` (115), `ReplayRetriever` (141), `RetrievalSnapshot` (155), `QuestionOutcome` (166), `GateResult` (530), `_schema_key` (57), `_band` (204), and the constants `PINNED_REVIEW_FLOOR`, `PINNED_THRESHOLD`, `FULL_REPEATS`, `CANARY_REPEATS`, `CANARY_MUST_ANSWER`, `CANARY_MUST_REFUSE`, `_POLICIES`, `_PHYSICAL_EVENTS`. No I/O, no report writing.
- `runner.py` — `capture_snapshots` (212), `_use_case` (245), `_lang` (258), `_run_question` (264), `run_matrix` (316), `_assert_profile_coverage` (882), `_canary_questions` (904), `_Prereqs` (860), `_verify_prereqs` (865). Owns the LLM and retrieval calls and the single owning `asyncio.run`.
- `artifacts.py` — `_CHECKLIST_HEADER` (357), `_EDITABLE_COLUMNS`/`_VERDICT_COLUMNS`/`_IMMUTABLE_COLUMNS` (375–377), `_arm_labels` (380), `_checklist_rows` (388), `_write_checklist` (427), `checklist_baseline` (434), `_percentiles` (717), `_sha256_file` (727), `_write_jsonl` (731), `render_comparison` (737), `_finalize_dir` (805), `write_run_dir` (815), and `REPORT_ROOT` (37).
- `verdicts.py` — `HumanVerdicts` (442), `_parse_pass` (450), `_verify_against_baseline` (459), `import_human_verdicts` (488), `_rate` (536), `evaluate_gates` (545), `import_verdicts_into_run` (1025).
- `src/features/evaluation/gate_generation_eval.py` — reduced to `run` (922), `main` (1126), and re-exports of every public name above. `python -m src.features.evaluation.gate_generation_eval` and `--import-verdicts` keep working unchanged.

Tests: `tests/test_evaluation_gate_generation_eval.py` keeps its module-level `import src.features.evaluation.gate_generation_eval as gge` and needs **no** rewrite — that is the point of the façade. `tests/test_gate_eval_artifact_contract.py` is new.

---

### Task 1: Freeze the artifact contract before touching anything

**Files:**
- Create: `tests/test_gate_eval_artifact_contract.py`, `tests/fixtures/gate_eval_expected/` (generated in Step 3)
- Test: `tests/test_gate_eval_artifact_contract.py`

**Interfaces:**
- Consumes: the injected-fakes end-to-end path `tests/test_evaluation_gate_generation_eval.py::test_run_end_to_end_with_injected_fakes` already exercises (line 493).
- Produces: `tests/fixtures/gate_eval_expected/` holding the exact filenames, the checklist header row, and the sorted JSON key sets of `run_manifest.json` and one `outcomes.jsonl` row — the invariants a refactor could break silently. It deliberately does **not** freeze timestamps, run ids, latencies or checksum values, which are legitimately run-dependent.

- [ ] **Step 1: Write the failing test** — create `tests/test_gate_eval_artifact_contract.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "gate_eval_expected"


def _run_dir_with_fakes(tmp_path: Path) -> Path:
    """Reuses the injected-fakes end-to-end path — no provider call, no cost.
    Import the helpers from the existing suite rather than duplicating them."""
    from tests.test_evaluation_gate_generation_eval import build_fake_run  # see Step 2

    return build_fake_run(tmp_path)


def test_run_directory_filenames_are_frozen(tmp_path: Path) -> None:
    """These filenames are the run's public contract: the owner grades
    blind_checklist.csv by hand and re-verifies against checksums.txt. A
    refactor that renames one silently breaks an already-completed run."""
    produced = sorted(p.name for p in _run_dir_with_fakes(tmp_path).iterdir())
    expected = json.loads((FIXTURE_DIR / "filenames.json").read_text(encoding="utf-8"))

    assert produced == expected


def test_checklist_header_is_frozen(tmp_path: Path) -> None:
    path = _run_dir_with_fakes(tmp_path) / "blind_checklist.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))

    expected = json.loads((FIXTURE_DIR / "checklist_header.json").read_text(encoding="utf-8"))

    assert header == expected


def test_manifest_and_outcome_key_sets_are_frozen(tmp_path: Path) -> None:
    run_dir = _run_dir_with_fakes(tmp_path)
    manifest_keys = sorted(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")))
    first_outcome = json.loads((run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0])

    expected = json.loads((FIXTURE_DIR / "key_sets.json").read_text(encoding="utf-8"))

    assert manifest_keys == expected["run_manifest"]
    assert sorted(first_outcome) == expected["outcome_row"]


def test_a_completed_run_still_reverifies_after_the_refactor(tmp_path: Path) -> None:
    """checksums.txt covers the immutable half. If a move changed a byte in
    run_manifest.json or outcomes.jsonl, this is where it shows up."""
    import src.features.evaluation.gate_generation_eval as gge

    run_dir = _run_dir_with_fakes(tmp_path)
    recorded = dict(
        line.split("  ", 1)[::-1]
        for line in (run_dir / "checksums.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    for name, digest in recorded.items():
        assert gge._sha256_file(run_dir / name.strip()) == digest.strip()
```

Confirm `checksums.txt`'s actual line format with `sed -n '815,854p' src/features/evaluation/gate_generation_eval.py` and adjust the parsing in the last test to match it exactly rather than assuming `"<digest>  <name>"`.

- [ ] **Step 2: Expose a reusable fake-run builder**

`tests/test_evaluation_gate_generation_eval.py:493`'s `test_run_end_to_end_with_injected_fakes` already constructs everything needed. Extract its setup into a module-level `build_fake_run(tmp_path: Path) -> Path` in that same file and have the existing test call it, so there is exactly one definition of "a fake run" and the contract test cannot drift from the behaviour test.

- [ ] **Step 3: Run the test to verify it fails, then generate the fixture**

Run: `pytest tests/test_gate_eval_artifact_contract.py -q`

Expected: FAIL — `FileNotFoundError` on `tests/fixtures/gate_eval_expected/`.

Generate the fixture from the **current, pre-refactor** code — this is the baseline the whole bucket is measured against:

```bash
python - <<'PY'
import json, csv, tempfile
from pathlib import Path
from tests.test_evaluation_gate_generation_eval import build_fake_run

out = Path("tests/fixtures/gate_eval_expected"); out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    run_dir = build_fake_run(Path(tmp))
    (out / "filenames.json").write_text(json.dumps(sorted(p.name for p in run_dir.iterdir()), indent=2) + "\n", encoding="utf-8")
    with (run_dir / "blind_checklist.csv").open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    (out / "checklist_header.json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")
    manifest = sorted(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")))
    outcome = sorted(json.loads((run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0]))
    (out / "key_sets.json").write_text(json.dumps({"run_manifest": manifest, "outcome_row": outcome}, indent=2) + "\n", encoding="utf-8")
print("wrote", sorted(p.name for p in out.iterdir()))
PY
```

Read the three generated files before proceeding. If `filenames.json` does not list all nine expected artifacts, the fake-run builder is not exercising the full path and the fixture is worthless — fix that first.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_gate_eval_artifact_contract.py -q`

Expected: PASS against the un-refactored module. From here on, this file is the tripwire.

- [ ] **Step 5: Re-run an existing sealed run for extra confidence**

An untracked real run exists at `eval/reports/gate_generation_eval_20260902T225839Z/`. Verify it against the current code before refactoring:

```bash
python -c "import src.features.evaluation.gate_generation_eval as g, pathlib, sys; d=pathlib.Path('eval/reports/gate_generation_eval_20260902T225839Z'); print(sorted(p.name for p in d.iterdir()))"
```

Note the filename list. After every subsequent task, that same command must print the same list and the artifact-contract test must stay green. Do **not** modify, re-import, or overwrite anything inside that directory — it is a sealed, write-once run.

---

### Task 2: Extract `gate_eval/models.py`

**Files:**
- Create: `src/features/evaluation/gate_eval/__init__.py`, `src/features/evaluation/gate_eval/models.py`
- Modify: `src/features/evaluation/gate_generation_eval.py`
- Test: `tests/test_evaluation_gate_generation_eval.py`, `tests/test_gate_eval_artifact_contract.py`

**Interfaces:**
- Produces: `gate_eval.models.{TraceCollector, WithinRepeatCache, ReplayRetriever, RetrievalSnapshot, QuestionOutcome, GateResult, schema_key, band}` plus `PINNED_REVIEW_FLOOR`, `PINNED_THRESHOLD`, `FULL_REPEATS`, `CANARY_REPEATS`, `CANARY_MUST_ANSWER`, `CANARY_MUST_REFUSE`, `POLICIES`, `PHYSICAL_EVENTS`.
- `gate_generation_eval` re-exports all of them under their **current names**, including the underscore-prefixed `_schema_key`, `_band`, `_POLICIES`, `_PHYSICAL_EVENTS`, because `tests/test_evaluation_gate_generation_eval.py` reaches for `gge._POLICIES` and friends directly.

- [ ] **Step 1: Write the failing test** — append to `tests/test_evaluation_gate_generation_eval.py`:

```python
def test_models_live_in_their_own_module_and_stay_reachable_from_the_facade() -> None:
    """The façade is the compatibility contract: the CLI, this suite, and any
    sealed-run tooling keep importing gate_generation_eval."""
    from src.features.evaluation.gate_eval import models

    for name in (
        "TraceCollector",
        "WithinRepeatCache",
        "ReplayRetriever",
        "RetrievalSnapshot",
        "QuestionOutcome",
        "GateResult",
    ):
        assert getattr(models, name) is getattr(gge, name)

    assert gge.FULL_REPEATS == models.FULL_REPEATS == 3
    assert gge.CANARY_REPEATS == models.CANARY_REPEATS == 3
    assert gge.PINNED_THRESHOLD == 0.5999
    assert gge.PINNED_REVIEW_FLOOR == 0.5500
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_evaluation_gate_generation_eval.py -q -k models_live`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.features.evaluation.gate_eval'`.

- [ ] **Step 3: Move the code**

Create the package with an empty `__init__.py`, then move — **verbatim, docstrings included** — lines 44–50, 54, 57–59, 61–203 and 529–535 into `models.py`, renaming `_schema_key` → `schema_key`, `_band` → `band`, `_POLICIES` → `POLICIES`, `_PHYSICAL_EVENTS` → `PHYSICAL_EVENTS`. Carry over the imports each moved symbol needs (`dataclass`, `Optional`, `Any`, `LlmTraceEvent`, `RetrievalResult`, `DEFAULT_REFUSAL_COSINE_THRESHOLD`, `DEFAULT_REFUSAL_REVIEW_FLOOR`).

`WithinRepeatCache.generate_structured` moves as-is. If Bucket 2 has already landed, it has three parameters here, not four — do not "fix" that in this bucket.

- [ ] **Step 4: Add the façade re-exports**

At the top of `gate_generation_eval.py`:

```python
from src.features.evaluation.gate_eval.models import (
    CANARY_MUST_ANSWER,
    CANARY_MUST_REFUSE,
    CANARY_REPEATS,
    FULL_REPEATS,
    PHYSICAL_EVENTS as _PHYSICAL_EVENTS,
    PINNED_REVIEW_FLOOR,
    PINNED_THRESHOLD,
    POLICIES as _POLICIES,
    GateResult,
    QuestionOutcome,
    ReplayRetriever,
    RetrievalSnapshot,
    TraceCollector,
    WithinRepeatCache,
    band as _band,
    schema_key as _schema_key,
)
```

and delete the moved definitions. Keep a module-level `__all__` listing the public names so ruff does not flag the re-exports.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_evaluation_gate_generation_eval.py tests/test_gate_eval_artifact_contract.py -q
python -c "import pathlib; d=pathlib.Path('eval/reports/gate_generation_eval_20260902T225839Z'); print(sorted(p.name for p in d.iterdir()))"
```

Expected: PASS, and the sealed run's filename list unchanged from Task 1 Step 5.

---

### Task 3: Extract `gate_eval/artifacts.py`

**Files:**
- Create: `src/features/evaluation/gate_eval/artifacts.py`
- Modify: `src/features/evaluation/gate_generation_eval.py`
- Test: `tests/test_evaluation_gate_generation_eval.py`, `tests/test_gate_eval_artifact_contract.py`

**Interfaces:**
- Produces: `gate_eval.artifacts.{REPORT_ROOT, CHECKLIST_HEADER, EDITABLE_COLUMNS, VERDICT_COLUMNS, IMMUTABLE_COLUMNS, arm_labels, checklist_rows, write_checklist, checklist_baseline, percentiles, sha256_file, write_jsonl, render_comparison, finalize_dir, write_run_dir}`.
- The façade re-exports each under its current name (`_CHECKLIST_HEADER`, `_arm_labels`, `_sha256_file`, `_finalize_dir`, …). `REPORT_ROOT` must stay importable from `gate_generation_eval` — `CLAUDE.md` and the CLI both reference it.

**Naming collision to avoid:** `src/features/evaluation/artifacts.py` already exists (127 lines, provenance headers). The new module is `src/features/evaluation/gate_eval/artifacts.py` — a different package. Import the old one as `from src.features.evaluation import artifacts as provenance_artifacts` anywhere both are needed, and never use a bare `import artifacts`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_evaluation_gate_generation_eval.py`:

```python
def test_artifact_writers_live_in_gate_eval_artifacts() -> None:
    from src.features.evaluation.gate_eval import artifacts as gate_artifacts

    assert gate_artifacts.write_run_dir is gge.write_run_dir
    assert gate_artifacts.CHECKLIST_HEADER == gge._CHECKLIST_HEADER
    assert gate_artifacts.IMMUTABLE_COLUMNS == gge._IMMUTABLE_COLUMNS
    assert gge.REPORT_ROOT == gate_artifacts.REPORT_ROOT


def test_editable_and_immutable_columns_still_partition_the_header() -> None:
    """The graded CSV's safety property: the owner may edit only the four
    verdict/notes columns, and _verify_against_baseline rejects any change to
    the rest. A split that dropped a column from IMMUTABLE_COLUMNS would make
    a tampered row pass."""
    from src.features.evaluation.gate_eval import artifacts as gate_artifacts

    assert set(gate_artifacts.EDITABLE_COLUMNS) | set(gate_artifacts.IMMUTABLE_COLUMNS) == set(
        gate_artifacts.CHECKLIST_HEADER
    )
    assert not set(gate_artifacts.EDITABLE_COLUMNS) & set(gate_artifacts.IMMUTABLE_COLUMNS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_gate_generation_eval.py -q -k "gate_eval_artifacts or partition"`

Expected: FAIL — `ModuleNotFoundError` on `src.features.evaluation.gate_eval.artifacts`.

- [ ] **Step 3: Move the code**

Move lines 37, 357–440 and 712–853 verbatim into `gate_eval/artifacts.py`, dropping the leading underscores on the names listed in the Interfaces block. Carry the imports: `csv`, `json`, `hashlib`, `shutil`/`os` (whatever `_finalize_dir` uses), `Path`, `Any`, and `from src.features.evaluation.gate_eval.models import QuestionOutcome, RetrievalSnapshot, GateResult`.

If Bucket 1 Task 3 has landed, `REPORT_ROOT = EVAL_REPORTS_DIR` from `src.core.paths`; otherwise keep the existing `Path(__file__)...` chain, adjusting the parent count for the extra package level — `gate_eval/artifacts.py` is one level deeper than the old file, so it needs **five** `.parent` hops, not four. This off-by-one is exactly what Bucket 1 Task 3 exists to eliminate; landing that bucket first avoids the trap entirely.

- [ ] **Step 4: Add the façade re-exports**

```python
from src.features.evaluation.gate_eval.artifacts import (
    CHECKLIST_HEADER as _CHECKLIST_HEADER,
    EDITABLE_COLUMNS as _EDITABLE_COLUMNS,
    IMMUTABLE_COLUMNS as _IMMUTABLE_COLUMNS,
    REPORT_ROOT,
    VERDICT_COLUMNS as _VERDICT_COLUMNS,
    arm_labels as _arm_labels,
    checklist_baseline,
    checklist_rows as _checklist_rows,
    finalize_dir as _finalize_dir,
    percentiles as _percentiles,
    render_comparison,
    sha256_file as _sha256_file,
    write_checklist as _write_checklist,
    write_jsonl as _write_jsonl,
    write_run_dir,
)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_evaluation_gate_generation_eval.py tests/test_gate_eval_artifact_contract.py -q
```

Expected: PASS, with the artifact-contract test confirming filenames, checklist header, key sets and checksum re-verification are all unchanged. A failure here is a real byte-level regression — fix the move, do not regenerate the fixture.

---

### Task 4: Extract `gate_eval/runner.py` and `gate_eval/verdicts.py`, leaving the façade

**Files:**
- Create: `src/features/evaluation/gate_eval/runner.py`, `src/features/evaluation/gate_eval/verdicts.py`
- Modify: `src/features/evaluation/gate_generation_eval.py` (reduced to `run`, `main`, re-exports), `CLAUDE.md`
- Test: `tests/test_evaluation_gate_generation_eval.py`, `tests/test_gate_eval_artifact_contract.py`

**Interfaces:**
- Produces: `gate_eval.runner.{capture_snapshots, run_matrix, use_case, lang, run_question, assert_profile_coverage, canary_questions, Prereqs, verify_prereqs}` and `gate_eval.verdicts.{HumanVerdicts, parse_pass, verify_against_baseline, import_human_verdicts, rate, evaluate_gates, import_verdicts_into_run}`.
- `gate_generation_eval` keeps `run` and `main` (they wire everything together and own the CLI surface) plus re-exports of all of the above under today's names.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_evaluation_gate_generation_eval.py`:

```python
def test_runner_and_verdicts_live_in_their_own_modules() -> None:
    from src.features.evaluation.gate_eval import runner, verdicts

    assert runner.run_matrix is gge.run_matrix
    assert runner.capture_snapshots is gge.capture_snapshots
    assert verdicts.evaluate_gates is gge.evaluate_gates
    assert verdicts.import_verdicts_into_run is gge.import_verdicts_into_run


def test_facade_is_small_enough_to_hold_in_one_head() -> None:
    """The audit's finding was a 1143-line module. The façade should be a wiring
    file: run(), main(), and re-exports."""
    from pathlib import Path

    module = Path(gge.__file__)
    assert len(module.read_text(encoding="utf-8").splitlines()) < 250


def test_no_gate_eval_module_exceeds_the_split_budget() -> None:
    from pathlib import Path

    package = Path(gge.__file__).parent / "gate_eval"
    oversized = {
        p.name: len(p.read_text(encoding="utf-8").splitlines())
        for p in package.glob("*.py")
        if len(p.read_text(encoding="utf-8").splitlines()) > 450
    }
    assert not oversized, f"gate_eval modules still too large: {oversized}"
```

The two size thresholds are budgets, not sacred numbers. If a module lands slightly over because a docstring moved with it, raise the budget in the test and say so — do not shave a docstring to fit.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_gate_generation_eval.py -q -k "own_modules or one_head or split_budget"`

Expected: FAIL — the modules do not exist and the façade is still ~900 lines.

- [ ] **Step 3: Move the runner**

Move lines 212–352 and 859–921 verbatim into `runner.py`, de-underscoring `_use_case`, `_lang`, `_run_question`, `_assert_profile_coverage`, `_canary_questions`, `_Prereqs`, `_verify_prereqs`. It imports from `gate_eval.models`, from `src.features.evaluation._eval_retriever`, and from the frozen-dataset integrity modules — carry each import across.

`run_matrix`'s single owning `asyncio.run` and its per-repeat close stay exactly as they are. So does the rule that the confident-band LLM call is shared **within** a repeat and never across one: that is `WithinRepeatCache`'s contract and it lives in `models.py` now.

- [ ] **Step 4: Move the verdicts layer, then reduce the façade**

Move lines 441–523 and 529–711 and 1025–1125 into `verdicts.py`, de-underscoring `_parse_pass`, `_verify_against_baseline`, `_rate`. `evaluate_gates` keeps its exact signature — including the sealed `expected_holdout_ids` / `expected_canary_ids` parameters PR #9 added — and its gate list, order and detail strings are untouched.

`import_verdicts_into_run` keeps: verify immutables against `checksums.txt` first, then the graded CSV against the sealed baseline, then the answered-unanswerable cross-check against `outcomes.jsonl`, then write `import_manifest.json` / `comparison.import.md` / `checksums.import.txt` as **separate** artifacts, failing closed if `checksums.import.txt` already exists.

`gate_generation_eval.py` then holds `run` (922–1024), `main` (1126–end), and the re-export block. Add a module docstring saying it is the compatibility façade and that the implementation lives in `gate_eval/`.

- [ ] **Step 5: Update `CLAUDE.md` and run the full bucket verification**

In `CLAUDE.md`'s `gate_generation_eval` command entry, add one sentence: the implementation now lives in `src/features/evaluation/gate_eval/` (`models`, `runner`, `artifacts`, `verdicts`) and `gate_generation_eval.py` is the compatibility façade holding `run`/`main`; the CLI surface, the artifact contract and every gate are unchanged. Everything else in that entry stays true as written.

```bash
pytest -q
ruff check src tests
mypy src
python -c "import pathlib; d=pathlib.Path('eval/reports/gate_generation_eval_20260902T225839Z'); print(sorted(p.name for p in d.iterdir()))"
python -m src.features.evaluation.gate_holdout_integrity --verify
```

Expected: all green; the sealed run's filename list identical to Task 1 Step 5; `gate_generation_eval --help` still works (`python -m src.features.evaluation.gate_generation_eval --help`). Stop at green — committing needs the owner's separate authorization, and no paid run is part of this bucket.
