from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features.evaluation import eval_set_integrity, gate_holdout_integrity, regression_set_integrity

_KNOWN_CHUNK_IDS = [f"doc-a::chunk-{i:04d}" for i in range(12)]


def _chunks_file(tmp_path: Path) -> Path:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "".join(json.dumps({"chunk_id": cid}) + "\n" for cid in _KNOWN_CHUNK_IDS),
        encoding="utf-8",
    )
    return path


def _eval_set_file(tmp_path: Path, questions: list[dict] | None = None) -> Path:
    questions = questions if questions is not None else [{"id": "e1", "question": "an unrelated eval question"}]
    path = tmp_path / "eval_set.json"
    payload = {"version": "test", "sha256": eval_set_integrity.compute_hash(questions), "questions": questions}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _regression_file(tmp_path: Path, queries: list[dict] | None = None) -> Path:
    queries = queries if queries is not None else [{"id": "r1", "query": "an unrelated regression query"}]
    path = tmp_path / "regression_queries.json"
    payload = {"version": "test", "sha256": regression_set_integrity.compute_hash(queries), "queries": queries}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _verify_kwargs(tmp_path: Path) -> dict:
    return {
        "chunks_path": _chunks_file(tmp_path),
        "eval_set_path": _eval_set_file(tmp_path),
        "regression_set_path": _regression_file(tmp_path),
    }


def _valid_questions() -> list[dict]:
    questions: list[dict] = []
    for intent in range(12):
        for lang in ("en", "es"):
            questions.append(
                {
                    "id": f"a{intent:02d}-{lang}",
                    "pair_id": f"a{intent:02d}",
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
                    "pair_id": f"u{intent:02d}",
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
    gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_committed_holdout_is_still_a_draft_and_is_rejected():
    """The shipped eval/gate_holdout_v1.0.0.json is an empty draft on purpose;
    the owner must author + freeze the 48 questions. CI is red on the dedicated
    holdout step by design until that happens."""
    with pytest.raises(ValueError, match="frozen"):
        gate_holdout_integrity.verify()


def test_verify_rejects_hash_drift(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions())
    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["questions"][0]["question"] = "tampered"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_requires_chunks_file(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions())
    with pytest.raises(ValueError, match="chunks.jsonl|not found"):
        gate_holdout_integrity.verify(
            path,
            chunks_path=tmp_path / "missing.jsonl",
            eval_set_path=tmp_path / "no_eval.json",
            regression_set_path=tmp_path / "no_regression.json",
        )


def test_verify_rejects_wrong_total(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions()[:47])
    with pytest.raises(ValueError, match="exactly 48"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_duplicate_ids(tmp_path: Path):
    questions = _valid_questions()
    questions[1]["id"] = questions[0]["id"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="duplicate ids"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_missing_pair_id(tmp_path: Path):
    questions = _valid_questions()
    del questions[0]["pair_id"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="pair_id"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_pair_without_both_languages(tmp_path: Path):
    questions = _valid_questions()
    questions[1]["language"] = "en"  # a00 now has two EN, no ES
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="EN \\+ one ES|EN / "):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_pair_with_mismatched_answerable(tmp_path: Path):
    questions = _valid_questions()
    questions[1]["answerable"] = False
    questions[1]["absence_note"] = "flipped one half of the pair"
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="answerable"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_language_imbalance(tmp_path: Path):
    questions = _valid_questions()
    questions[0]["language"] = "es"
    questions[0]["pair_id"] = "loose-a"
    questions[1]["pair_id"] = "loose-b"
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="EN / |pair_id"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_answerable_missing_expected_chunks(tmp_path: Path):
    questions = _valid_questions()
    del questions[0]["expected_chunk_ids"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="expected_chunk_ids"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_expected_chunk_id_not_in_corpus(tmp_path: Path):
    questions = _valid_questions()
    questions[0]["expected_chunk_ids"] = ["doc-a::chunk-9999"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="not in chunks.jsonl"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_unanswerable_missing_absence_note(tmp_path: Path):
    questions = _valid_questions()
    del questions[24]["absence_note"]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="absence_note"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_question_colliding_with_eval_set(tmp_path: Path):
    questions = _valid_questions()
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="collides"):
        gate_holdout_integrity.verify(
            path,
            chunks_path=_chunks_file(tmp_path),
            eval_set_path=_eval_set_file(
                tmp_path, questions=[{"id": "e1", "question": "Answerable Intent 0   (EN)"}]
            ),
            regression_set_path=_regression_file(tmp_path),
        )


def test_verify_rejects_question_colliding_with_regression_set(tmp_path: Path):
    questions = _valid_questions()
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="collides"):
        gate_holdout_integrity.verify(
            path,
            chunks_path=_chunks_file(tmp_path),
            eval_set_path=_eval_set_file(tmp_path),
            regression_set_path=_regression_file(
                tmp_path, queries=[{"id": "r1", "query": "unanswerable intent 3 (es)"}]
            ),
        )


def test_verify_requires_eval_set_present(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions())
    with pytest.raises(ValueError, match="eval_set.json not found"):
        gate_holdout_integrity.verify(
            path,
            chunks_path=_chunks_file(tmp_path),
            eval_set_path=tmp_path / "missing_eval.json",
            regression_set_path=_regression_file(tmp_path),
        )


def test_verify_requires_regression_set_present(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions())
    with pytest.raises(ValueError, match="regression_queries.json not found"):
        gate_holdout_integrity.verify(
            path,
            chunks_path=_chunks_file(tmp_path),
            eval_set_path=_eval_set_file(tmp_path),
            regression_set_path=tmp_path / "missing_regression.json",
        )


def test_verify_rejects_tampered_eval_set(tmp_path: Path):
    path = _frozen_holdout(tmp_path, _valid_questions())
    eval_set = _eval_set_file(tmp_path)
    payload = json.loads(eval_set.read_text(encoding="utf-8"))
    payload["questions"][0]["question"] = "silently changed after freezing"
    eval_set.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gate_holdout_integrity.verify(
            path,
            chunks_path=_chunks_file(tmp_path),
            eval_set_path=eval_set,
            regression_set_path=_regression_file(tmp_path),
        )


def test_verify_rejects_internal_normalized_duplicate(tmp_path: Path):
    questions = _valid_questions()
    questions[2]["question"] = "Answerable   Intent 0 (EN)"  # normalizes to questions[0]
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="identical after normalization"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_verify_rejects_answerable_pair_with_mismatched_expected_chunks(tmp_path: Path):
    questions = _valid_questions()
    questions[1]["expected_chunk_ids"] = [_KNOWN_CHUNK_IDS[5]]  # ES half of pair a00 now differs from EN
    path = _frozen_holdout(tmp_path, questions)
    with pytest.raises(ValueError, match="same expected_chunk_ids"):
        gate_holdout_integrity.verify(path, **_verify_kwargs(tmp_path))


def test_write_is_atomic_and_leaves_no_tmp_file(tmp_path: Path):
    questions = _valid_questions()
    path = tmp_path / "gate_holdout.json"
    path.write_text(
        json.dumps({"version": "1.0.0", "sha256": "stale", "status": "frozen", "questions": questions}),
        encoding="utf-8",
    )
    gate_holdout_integrity.write(path)
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["sha256"] == gate_holdout_integrity.compute_hash(
        questions
    )
