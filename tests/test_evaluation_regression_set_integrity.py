from __future__ import annotations

import json

from src.features.evaluation.eval_set_integrity import EVAL_SET_FILE
from src.features.evaluation.regression_set_integrity import (
    REGRESSION_SET_FILE,
    load_regression_set,
    verify,
)

_CHUNKS_FILE = EVAL_SET_FILE.resolve().parent.parent / "ingestion" / "output" / "chunks.jsonl"


def _known_chunk_ids() -> set[str]:
    with _CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f}


def test_regression_set_file_stays_at_eval_directory_not_under_src():
    assert REGRESSION_SET_FILE.parent.name == "eval"
    assert "src" not in REGRESSION_SET_FILE.parts


def test_regression_set_hash_is_frozen():
    verify()


def test_regression_rows_are_well_formed():
    data = load_regression_set()
    ids = [q["id"] for q in data["queries"]]
    assert len(ids) == len(set(ids))
    known = _known_chunk_ids()
    for q in data["queries"]:
        assert q["language"] in ("en", "es")
        assert isinstance(q["should_answer"], bool)
        if q["should_answer"]:
            assert q["expected_chunk_id"] in known, f"{q['id']}: {q['expected_chunk_id']!r} not in chunks.jsonl"
        else:
            assert q["expected_chunk_id"] is None


def test_regression_set_has_language_pairs_and_controls():
    data = load_regression_set()
    assert sum(1 for q in data["queries"] if q["language"] == "es") >= 6
    assert sum(1 for q in data["queries"] if not q["should_answer"]) >= 2
