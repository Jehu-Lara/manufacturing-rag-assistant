from __future__ import annotations

import csv
import json
from pathlib import Path

import src.features.evaluation.gate_generation_eval as gge
from tests.test_evaluation_gate_generation_eval import build_fake_run

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "gate_eval_expected"


def test_run_directory_filenames_are_frozen(tmp_path: Path) -> None:
    """These filenames are the run's public contract: the owner grades
    blind_checklist.csv by hand and re-verifies against checksums.txt. A
    refactor that renames one silently breaks an already-completed run."""
    produced = sorted(p.name for p in build_fake_run(tmp_path).iterdir())
    expected = json.loads((FIXTURE_DIR / "filenames.json").read_text(encoding="utf-8"))

    assert produced == expected


def test_checklist_header_is_frozen(tmp_path: Path) -> None:
    path = build_fake_run(tmp_path) / "blind_checklist.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))

    expected = json.loads((FIXTURE_DIR / "checklist_header.json").read_text(encoding="utf-8"))

    assert header == expected


def test_manifest_and_outcome_key_sets_are_frozen(tmp_path: Path) -> None:
    run_dir = build_fake_run(tmp_path)
    manifest_keys = sorted(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")))
    first_outcome = json.loads(
        (run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    expected = json.loads((FIXTURE_DIR / "key_sets.json").read_text(encoding="utf-8"))

    assert manifest_keys == expected["run_manifest"]
    assert sorted(first_outcome) == expected["outcome_row"]


def test_a_completed_run_still_reverifies_after_the_refactor(tmp_path: Path) -> None:
    """checksums.txt covers the immutable half of the run. If a move changed a
    byte in run_manifest.json or outcomes.jsonl, this is where it shows up."""
    run_dir = build_fake_run(tmp_path)
    lines = (run_dir / "checksums.txt").read_text(encoding="utf-8").splitlines()

    recorded = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        digest, name = stripped.split(maxsplit=1)
        recorded[name] = digest

    assert recorded, "checksums.txt is empty"
    for name, digest in recorded.items():
        assert gge._sha256_file(run_dir / name) == digest


def test_implementation_lives_in_gate_eval_and_stays_reachable_from_the_facade() -> None:
    """The façade is the compatibility contract: the CLI, this suite, and any
    sealed-run tooling keep importing gate_generation_eval."""
    from src.features.evaluation.gate_eval import artifacts, models, runner, verdicts

    assert models.WithinRepeatCache is gge.WithinRepeatCache
    assert models.QuestionOutcome is gge.QuestionOutcome
    assert models.GateResult is gge.GateResult
    assert runner.run_matrix is gge.run_matrix
    assert runner.capture_snapshots is gge.capture_snapshots
    assert artifacts.write_run_dir is gge.write_run_dir
    assert artifacts._CHECKLIST_HEADER == gge._CHECKLIST_HEADER
    assert verdicts.evaluate_gates is gge.evaluate_gates
    assert verdicts.import_human_verdicts is gge.import_human_verdicts

    assert gge.FULL_REPEATS == 3
    assert gge.CANARY_REPEATS == 3
    assert gge.PINNED_THRESHOLD == 0.5999
    assert gge.PINNED_REVIEW_FLOOR == 0.5500


def test_editable_and_immutable_columns_still_partition_the_header() -> None:
    """The graded CSV's safety property: the owner may edit only the verdict
    and notes columns, and _verify_against_baseline rejects any change to the
    rest. A split that dropped a column from _IMMUTABLE_COLUMNS would let a
    tampered row pass."""
    from src.features.evaluation.gate_eval import artifacts

    editable = set(artifacts._EDITABLE_COLUMNS)
    immutable = set(artifacts._IMMUTABLE_COLUMNS)

    assert editable | immutable == set(artifacts._CHECKLIST_HEADER)
    assert not editable & immutable


def test_facade_is_a_wiring_file_and_no_module_is_oversized() -> None:
    """The audit's finding was one 1143-line module."""
    facade = Path(gge.__file__)
    assert len(facade.read_text(encoding="utf-8").splitlines()) < 500

    package = facade.parent / "gate_eval"
    oversized = {
        p.name: len(p.read_text(encoding="utf-8").splitlines())
        for p in package.glob("*.py")
        if len(p.read_text(encoding="utf-8").splitlines()) > 450
    }
    assert not oversized, f"gate_eval modules still too large: {oversized}"
