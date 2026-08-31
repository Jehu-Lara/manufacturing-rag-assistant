from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features.evaluation import gate_holdout_integrity


def test_committed_gate_holdout_hash_verifies():
    gate_holdout_integrity.verify()


def test_compute_hash_is_order_independent_of_keys():
    a = [{"id": "h1", "question": "x", "answerable": True}]
    b = [{"answerable": True, "question": "x", "id": "h1"}]
    assert gate_holdout_integrity.compute_hash(a) == gate_holdout_integrity.compute_hash(b)


def test_verify_raises_on_content_drift(tmp_path: Path):
    path = tmp_path / "gate_holdout.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "sha256": gate_holdout_integrity.compute_hash([{"id": "h1"}]),
                "questions": [{"id": "h1-mutated"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gate_holdout_integrity.verify(path)


def test_write_then_verify_roundtrips(tmp_path: Path):
    path = tmp_path / "gate_holdout.json"
    path.write_text(
        json.dumps(
            {"version": "1.0.0", "sha256": "", "questions": [{"id": "h1", "answerable": False}]}
        ),
        encoding="utf-8",
    )
    gate_holdout_integrity.write(path)
    gate_holdout_integrity.verify(path)
