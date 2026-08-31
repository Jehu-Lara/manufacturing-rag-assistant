from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features.evaluation import gate_holdout_integrity

_KNOWN_CHUNK_IDS = [f"doc-a::chunk-{i:04d}" for i in range(12)]


def _chunks_file(tmp_path: Path) -> Path:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "".join(json.dumps({"chunk_id": cid}) + "\n" for cid in _KNOWN_CHUNK_IDS),
        encoding="utf-8",
    )
    return path


def _valid_questions() -> list[dict]:
    questions: list[dict] = []
    for intent in range(12):
        for lang in ("en", "es"):
            questions.append(
                {
                    "id": f"a{intent:02d}-{lang}",
                    "question": f"answerable intent {intent} ({lang})",
                    "language": lang,
                    "answerable": True,
                    "expected_chunk_ids": [_KNOWN_CHUNK_IDS[intent]],
                    "expected_answer": f"the answer for intent {intent}",
                }
            )
    for intent in range(12):
        for lang in ("en", "es"):
            questions.append(
                {
                    "id": f"u{intent:02d}-{lang}",
                    "question": f"unanswerable intent {intent} ({lang})",
                    "language": lang,
                    "answerable": False,
                    "absence_note": f"corpus has no material on intent {intent}",
                }
            )
    return questions


def _frozen_holdout(tmp_path: Path, questions: list[dict]) -> Path:
    path = tmp_path / "gate_holdout.json"
    payload = {
        "version": "1.0.0",
        "sha256": gate_holdout_integrity.compute_hash(questions),
        "status": "frozen",
        "questions": questions,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compute_hash_is_key_order_independent():
    a = [{"id": "h1", "question": "x", "answerable": True}]
    b = [{"answerable": True, "question": "x", "id": "h1"}]
    assert gate_holdout_integrity.compute_hash(a) == gate_holdout_integrity.compute_hash(b)


def test_valid_frozen_48_question_holdout_verifies(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions())
    gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_committed_holdout_is_still_a_draft_and_is_rejected():
    """The shipped eval/gate_holdout_v1.0.0.json is an empty draft on purpose;
    the owner must author + freeze the 48 questions. CI is red here by design
    until that happens."""
    with pytest.raises(ValueError, match="frozen"):
        gate_holdout_integrity.verify()


def test_verify_rejects_hash_drift(tmp_path: Path):
    questions = _valid_questions()
    path = _frozen_holdout(tmp_path, questions)
    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["questions"][0]["question"] = "tampered"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_wrong_total(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions()[:47])
    with pytest.raises(ValueError, match="exactly 48"):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_duplicate_ids(tmp_path: Path):
    questions = _valid_questions()
    questions[1]["id"] = questions[0]["id"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="duplicate ids"):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_class_imbalance(tmp_path: Path):
    questions = _valid_questions()
    questions[0]["answerable"] = False
    questions[0]["absence_note"] = "flipped"
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="answerable / "):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_language_imbalance(tmp_path: Path):
    questions = _valid_questions()
    questions[0]["language"] = "es"
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="EN / "):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_answerable_missing_expected_chunks(tmp_path: Path):
    questions = _valid_questions()
    del questions[0]["expected_chunk_ids"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="expected_chunk_ids"):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_expected_chunk_id_not_in_corpus(tmp_path: Path):
    questions = _valid_questions()
    questions[0]["expected_chunk_ids"] = ["doc-a::chunk-9999"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="not in chunks.jsonl"):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))


def test_verify_rejects_unanswerable_missing_absence_note(tmp_path: Path):
    questions = _valid_questions()
    del questions[24]["absence_note"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="absence_note"):
        gate_holdout_integrity.verify(path, chunks_path=_chunks_file(tmp_path))
