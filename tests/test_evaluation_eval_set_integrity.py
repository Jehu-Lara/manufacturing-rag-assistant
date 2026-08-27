from __future__ import annotations

import json

from src.features.evaluation.eval_set_integrity import EVAL_SET_FILE, compute_hash, load_eval_set, verify

_CHUNKS_FILE = EVAL_SET_FILE.resolve().parent.parent / "ingestion" / "output" / "chunks.jsonl"


def _known_chunk_ids() -> set[str]:
    with _CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f}


def test_eval_set_file_stays_at_eval_directory_not_under_src():
    assert EVAL_SET_FILE.parent.name == "eval"
    assert "src" not in EVAL_SET_FILE.parts


def test_stored_hash_matches_computed_hash():
    verify()


def test_hash_is_sensitive_to_content_changes():
    data = load_eval_set()
    tampered = json.loads(json.dumps(data["questions"]))
    tampered[0]["question"] = tampered[0]["question"] + " (tampered)"
    assert compute_hash(tampered) != data["sha256"]


def test_eval_set_has_forty_items_with_declared_split():
    data = load_eval_set()
    questions = data["questions"]
    assert len(questions) == 40

    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    assert len(answerable) == 30
    assert len(unanswerable) == 10


def test_every_answerable_question_has_expected_chunk_ids_that_exist():
    data = load_eval_set()
    known_ids = _known_chunk_ids()
    for question in data["questions"]:
        if question["answerable"]:
            assert question["expected_chunk_ids"], f"{question['id']}: answerable but has no expected_chunk_ids"
            for chunk_id in question["expected_chunk_ids"]:
                assert chunk_id in known_ids, f"{question['id']}: expected_chunk_id {chunk_id!r} not in chunks.jsonl"


def test_unanswerable_questions_have_no_expected_chunk_ids():
    data = load_eval_set()
    for question in data["questions"]:
        if not question["answerable"]:
            assert question["expected_chunk_ids"] == []


def test_no_duplicate_question_ids():
    data = load_eval_set()
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids))


def test_spanish_questions_present_for_bilingual_validation():
    data = load_eval_set()
    spanish = [q for q in data["questions"] if q["language"] == "es"]
    assert 6 <= len(spanish) <= 8
    assert all(q["answerable"] for q in spanish), "Spanish subset is drawn from the answerable set"
