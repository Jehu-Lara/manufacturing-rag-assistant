from __future__ import annotations

import csv
import json
from unittest.mock import patch

from api.schemas import Citation, QueryResponse
from eval import hash_eval_set
from eval.run_generation_eval import (
    CSV_COLUMNS,
    INTER_QUESTION_DELAY_SECONDS,
    correct_refusal_rate,
    false_refusal_rate,
    run,
)
from retrieval.hybrid import RetrievalResult

THRESHOLD = 0.5599


def _row(answerable: bool, refused: bool) -> dict:
    return {"answerable": answerable, "refused": refused}


def test_correct_refusal_rate_all_correct():
    rows = [_row(False, True), _row(False, True), _row(True, False)]
    assert correct_refusal_rate(rows) == 1.0


def test_correct_refusal_rate_all_wrong():
    rows = [_row(False, False), _row(False, False)]
    assert correct_refusal_rate(rows) == 0.0


def test_correct_refusal_rate_mixed():
    rows = [_row(False, True), _row(False, False), _row(True, False)]
    assert correct_refusal_rate(rows) == 0.5


def test_correct_refusal_rate_empty_is_safe():
    assert correct_refusal_rate([]) == 0.0


def test_correct_refusal_rate_ignores_answerable_rows():
    rows = [_row(True, True), _row(True, True)]
    assert correct_refusal_rate(rows) == 0.0


def test_false_refusal_rate_all_correct():
    rows = [_row(True, False), _row(True, False), _row(False, True)]
    assert false_refusal_rate(rows) == 0.0


def test_false_refusal_rate_all_wrong():
    rows = [_row(True, True), _row(True, True)]
    assert false_refusal_rate(rows) == 1.0


def test_false_refusal_rate_mixed():
    rows = [_row(True, False), _row(True, True), _row(False, True)]
    assert false_refusal_rate(rows) == 0.5


def test_false_refusal_rate_empty_is_safe():
    assert false_refusal_rate([]) == 0.0


def test_false_refusal_rate_ignores_unanswerable_rows():
    rows = [_row(False, True), _row(False, True)]
    assert false_refusal_rate(rows) == 0.0


def _fake_questions() -> list[dict]:
    return [
        {
            "id": "fq001",
            "question": "answerable question one",
            "language": "en",
            "answerable": True,
            "expected_chunk_ids": ["doc-a::chunk-0001"],
            "expected_document_id": "doc-a",
            "expected_section_heading": "Section A",
            "expected_answer": "The answer to question one.",
            "notes": "fixture",
        },
        {
            "id": "fq002",
            "question": "answerable question two, falsely refused",
            "language": "es",
            "answerable": True,
            "expected_chunk_ids": ["doc-b::chunk-0002"],
            "expected_document_id": "doc-b",
            "expected_section_heading": "Section B",
            "expected_answer": "La respuesta a la pregunta dos.",
            "notes": "fixture",
        },
        {
            "id": "fq003",
            "question": "unanswerable question, correctly refused",
            "language": "en",
            "answerable": False,
            "expected_chunk_ids": [],
            "expected_document_id": None,
            "expected_section_heading": None,
            "expected_answer": "Not answerable from this corpus.",
            "notes": "fixture",
        },
        {
            "id": "fq004",
            "question": "unanswerable question, wrongly answered",
            "language": "en",
            "answerable": False,
            "expected_chunk_ids": [],
            "expected_document_id": None,
            "expected_section_heading": None,
            "expected_answer": "Not answerable from this corpus.",
            "notes": "fixture",
        },
    ]


def _write_fake_eval_set(path) -> dict:
    questions = _fake_questions()
    data = {
        "version": "9.9.9",
        "sha256": hash_eval_set.compute_hash(questions),
        "questions": questions,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _fake_retrieval_result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=1.0,
        semantic_rank=1,
        semantic_score=0.9,
        bm25_rank=1,
        bm25_score=1.0,
        metadata={
            "document_id": "doc-a",
            "document_title": "Doc A",
            "section_heading": "Section A",
            "revision": "Rev A",
            "chunk_id": chunk_id,
        },
    )


def _answer_question_side_effect(question: str, language: str):
    if question == "answerable question one":
        return QueryResponse(
            answer="The answer to question one.",
            citations=[
                Citation(
                    document_id="doc-a",
                    document_title="Doc A",
                    section_heading="Section A",
                    revision="Rev A",
                    chunk_id="doc-a::chunk-0001",
                )
            ],
            refused=False,
            status="ok",
            confidence=0.9,
            threshold=THRESHOLD,
            language=language,
            request_id="req-1",
        )
    if question == "answerable question two, falsely refused":
        return QueryResponse(
            answer="No puedo responder con confianza.",
            citations=[],
            refused=True,
            status="ok",
            confidence=0.2,
            threshold=THRESHOLD,
            language=language,
            request_id="req-2",
        )
    if question == "unanswerable question, correctly refused":
        return QueryResponse(
            answer="I cannot answer that from the provided sources.",
            citations=[],
            refused=True,
            status="ok",
            confidence=0.1,
            threshold=THRESHOLD,
            language=language,
            request_id="req-3",
        )
    if question == "unanswerable question, wrongly answered":
        return QueryResponse(
            answer="Here is a made-up answer.",
            citations=[],
            refused=False,
            status="ok",
            confidence=0.6,
            threshold=THRESHOLD,
            language=language,
            request_id="req-4",
        )
    raise AssertionError(f"unexpected question passed to answer_question: {question!r}")


def _retrieve_side_effect(query_text: str, k: int = 5):
    if query_text == "answerable question one":
        return [_fake_retrieval_result("doc-a::chunk-0001")]
    if query_text == "answerable question two, falsely refused":
        return [_fake_retrieval_result("doc-b::chunk-9999")]
    raise AssertionError(f"retrieve() should only be called for answerable questions, got: {query_text!r}")


@patch("time.sleep")
@patch("retrieval.hybrid.retrieve")
@patch("api.generation.answer_question")
def test_run_writes_report_and_csv_with_expected_metrics(mock_answer_question, mock_retrieve, mock_sleep, tmp_path):
    eval_set_path = tmp_path / "fake_eval_set.json"
    _write_fake_eval_set(eval_set_path)
    report_dir = tmp_path / "reports"

    mock_answer_question.side_effect = _answer_question_side_effect
    mock_retrieve.side_effect = _retrieve_side_effect

    report_path, csv_path = run(eval_set_path=eval_set_path, report_dir=report_dir)

    assert mock_answer_question.call_count == 4
    assert mock_retrieve.call_count == 2
    assert mock_sleep.call_count == 4
    mock_sleep.assert_called_with(INTER_QUESTION_DELAY_SECONDS)

    assert report_path == report_dir / "generation_eval_v9.9.9.md"
    assert csv_path == report_dir / "manual_review_checklist_v9.9.9.csv"
    assert report_path.exists()
    assert csv_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    lines = [line for line in report_text.splitlines() if line.strip()]
    assert lines[0] == "# Generation Evaluation Report — eval_set v9.9.9"
    assert lines[1] == "## Correct-Refusal Rate: 0.500"
    assert "False-refusal rate (answerable subset): 0.500" in report_text
    assert "fq001" in report_text
    assert "fq004" in report_text
    assert "manual_review_checklist_v9.9.9.csv" in report_text

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == 2
    ids = {row["id"] for row in rows}
    assert ids == {"fq001", "fq002"}

    row_one = next(row for row in rows if row["id"] == "fq001")
    assert row_one["language"] == "en"
    assert row_one["generated_answer"] == "The answer to question one."
    assert row_one["cited_document_ids"] == "doc-a"
    assert row_one["cited_section_headings"] == "Section A"
    assert row_one["retrieval_succeeded"] == "True"
    assert row_one["citation_accuracy_pass"] == ""
    assert row_one["faithfulness_pass"] == ""
    assert row_one["reviewer_notes"] == ""

    row_two = next(row for row in rows if row["id"] == "fq002")
    assert row_two["retrieval_succeeded"] == "False"
    assert row_two["cited_document_ids"] == ""
