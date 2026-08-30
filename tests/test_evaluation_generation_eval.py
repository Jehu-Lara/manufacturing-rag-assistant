from __future__ import annotations

import csv
import json
from unittest.mock import patch

import pytest

from src.core.config import Settings
from src.domain.models import RetrievalResult
from src.features.evaluation import eval_set_integrity
from src.features.evaluation.generation_eval import (
    CSV_COLUMNS,
    INTER_QUESTION_DELAY_SECONDS,
    correct_refusal_rate,
    false_refusal_rate,
    run,
)
from src.features.query.use_cases import QueryUseCase

THRESHOLD = 0.5599


class _FakeHeader:
    def render(self) -> str:
        return "## Provenance\n- index_profile: fake\n"


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
        "sha256": eval_set_integrity.compute_hash(questions),
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
            "chunk_text": "some chunk text",
        },
    )


class _ScriptedRetriever:
    """RetrieverPort fake dispatching by question text, mirroring the
    per-question fixtures below — retrieve() should only be called for
    answerable questions that survive to that point."""

    def __init__(self, by_question: dict[str, list[RetrievalResult]]) -> None:
        self._by_question = by_question

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        if query_text not in self._by_question:
            raise AssertionError(f"retrieve() should not have been called for: {query_text!r}")
        return self._by_question[query_text]


class _ScriptedLlmClient:
    """LLMClientPort fake — QueryUseCase never calls the LLM when the
    retriever's top-1 semantic_score is below threshold (0.9 here, always
    confident), so this always gets exercised for these fixtures."""

    def __init__(self, by_question: dict[str, dict]) -> None:
        self._by_question = by_question
        self._last_user_prompt: str = ""

    async def generate_structured(self, system_prompt, user_prompt, schema, settings):
        self._last_user_prompt = user_prompt
        for question_text, payload in self._by_question.items():
            if question_text in user_prompt:
                return payload
        raise AssertionError(f"no scripted LLM response matches prompt: {user_prompt!r}")


def _settings() -> Settings:
    return Settings(
        groq_api_key="fake",
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=THRESHOLD,
        log_level="INFO",
    )


def _build_fixtures():
    retriever = _ScriptedRetriever(
        {
            "answerable question one": [_fake_retrieval_result("doc-a::chunk-0001")],
            "answerable question two, falsely refused": [_fake_retrieval_result("doc-b::chunk-9999")],
            "unanswerable question, correctly refused": [],
            "unanswerable question, wrongly answered": [_fake_retrieval_result("doc-c::chunk-0003")],
        }
    )
    llm = _ScriptedLlmClient(
        {
            "answerable question one": {
                "answer": "The answer to question one.",
                "citations": [{"chunk_id": "doc-a::chunk-0001"}],
                "refused": False,
            },
            "answerable question two, falsely refused": {
                "answer": "No puedo responder con confianza.",
                "citations": [],
                "refused": True,
            },
            "unanswerable question, wrongly answered": {
                "answer": "Here is a made-up answer.",
                "citations": [{"chunk_id": "doc-c::chunk-0003"}],
                "refused": False,
            },
        }
    )
    use_case = QueryUseCase(retriever, llm, _settings())
    return use_case, retriever


@patch("src.features.evaluation.generation_eval.artifacts.resolve_provenance")
@patch("src.features.evaluation.generation_eval.time.sleep")
def test_run_writes_report_and_csv_with_expected_metrics(mock_sleep, mock_provenance, tmp_path):
    mock_provenance.return_value = _FakeHeader()
    eval_set_path = tmp_path / "fake_eval_set.json"
    _write_fake_eval_set(eval_set_path)
    report_dir = tmp_path / "reports"
    use_case, retriever = _build_fixtures()

    report_path, csv_path = run(
        eval_set_path=eval_set_path, report_dir=report_dir, use_case=use_case, retriever=retriever
    )

    assert mock_sleep.call_count == 4
    mock_sleep.assert_called_with(INTER_QUESTION_DELAY_SECONDS)

    assert report_path == report_dir / "generation_eval_v9.9.9__raw-v1__off.md"
    assert csv_path == report_dir / "manual_review_checklist_v9.9.9__raw-v1__off.csv"
    assert report_path.exists()
    assert csv_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    lines = [line for line in report_text.splitlines() if line.strip()]
    assert lines[0] == "## Provenance"
    assert "# Generation Evaluation Report — eval_set v9.9.9" in report_text
    assert "## Correct-Refusal Rate: 0.500" in report_text
    assert "False-refusal rate (answerable subset): 0.500" in report_text
    assert "fq001" in report_text
    assert "fq004" in report_text
    assert "manual_review_checklist_v9.9.9__raw-v1__off.csv" in report_text

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == 2
    assert {row["id"] for row in rows} == {"fq001", "fq002"}

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


@patch("src.features.evaluation.generation_eval.artifacts.resolve_provenance")
@patch("src.features.evaluation.generation_eval.time.sleep")
def test_run_off_and_semantic_produce_distinct_paths(mock_sleep, mock_provenance, tmp_path):
    mock_provenance.return_value = _FakeHeader()
    eval_set_path = tmp_path / "fake_eval_set.json"
    _write_fake_eval_set(eval_set_path)

    use_case, retriever = _build_fixtures()
    off_report, off_csv = run(
        eval_set_path=eval_set_path, report_dir=tmp_path / "r", use_case=use_case, retriever=retriever
    )
    use_case, retriever = _build_fixtures()
    sem_report, sem_csv = run(
        eval_set_path=eval_set_path,
        report_dir=tmp_path / "r",
        use_case=use_case,
        retriever=retriever,
        expansion_mode="semantic",
    )

    assert off_report != sem_report and off_csv != sem_csv
    assert off_report.name == "generation_eval_v9.9.9__raw-v1__off.md"
    assert sem_report.name == "generation_eval_v9.9.9__raw-v1__semantic.md"
    assert sem_csv.name == "manual_review_checklist_v9.9.9__raw-v1__semantic.csv"


@patch("src.features.evaluation.generation_eval.artifacts.resolve_provenance")
@patch("src.features.evaluation.generation_eval.time.sleep")
def test_run_canonical_alias_guarded(mock_sleep, mock_provenance, tmp_path):
    mock_provenance.return_value = _FakeHeader()
    eval_set_path = tmp_path / "fake_eval_set.json"
    _write_fake_eval_set(eval_set_path)

    use_case, retriever = _build_fixtures()
    with pytest.raises(ValueError):
        run(
            eval_set_path=eval_set_path,
            report_dir=tmp_path / "r",
            use_case=use_case,
            retriever=retriever,
            expansion_mode="semantic",
            write_canonical_alias=True,
        )

    use_case, retriever = _build_fixtures()
    report_path, _csv = run(
        eval_set_path=eval_set_path,
        report_dir=tmp_path / "r",
        use_case=use_case,
        retriever=retriever,
        write_canonical_alias=True,
    )
    assert report_path.name == "generation_eval_v9.9.9__raw-v1__off.md"
    assert (tmp_path / "r" / "generation_eval_v9.9.9.md").exists()
    assert (tmp_path / "r" / "manual_review_checklist_v9.9.9.csv").exists()


class _FailingThenScriptedRetriever(_ScriptedRetriever):
    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        if query_text == "answerable question one":
            raise RuntimeError("simulated transient network error")
        return super().retrieve(query_text, k, top_n)


class _FailingThenScriptedLlmClient(_ScriptedLlmClient):
    async def generate_structured(self, system_prompt, user_prompt, schema, settings):
        if "answerable question one" in user_prompt:
            raise RuntimeError("simulated transient network error")
        return await super().generate_structured(system_prompt, user_prompt, schema, settings)


@patch("src.features.evaluation.generation_eval.artifacts.resolve_provenance")
@patch("src.features.evaluation.generation_eval.time.sleep")
def test_run_continues_after_one_question_raises_and_records_error_row(
    mock_sleep, mock_provenance, tmp_path
):
    mock_provenance.return_value = _FakeHeader()
    eval_set_path = tmp_path / "fake_eval_set.json"
    _write_fake_eval_set(eval_set_path)
    report_dir = tmp_path / "reports"

    retriever = _FailingThenScriptedRetriever(
        {
            "answerable question two, falsely refused": [_fake_retrieval_result("doc-b::chunk-9999")],
            "unanswerable question, correctly refused": [],
            "unanswerable question, wrongly answered": [_fake_retrieval_result("doc-c::chunk-0003")],
        }
    )
    llm = _FailingThenScriptedLlmClient(
        {
            "answerable question two, falsely refused": {
                "answer": "No puedo responder con confianza.",
                "citations": [],
                "refused": True,
            },
            "unanswerable question, wrongly answered": {
                "answer": "Here is a made-up answer.",
                "citations": [{"chunk_id": "doc-c::chunk-0003"}],
                "refused": False,
            },
        }
    )
    use_case = QueryUseCase(retriever, llm, _settings())

    report_path, csv_path = run(
        eval_set_path=eval_set_path, report_dir=report_dir, use_case=use_case, retriever=retriever
    )

    assert report_path.exists()
    assert csv_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "| fq001 | en | True | False | error | n/a | n/a | n/a |" in report_text
    assert "fq002" in report_text
    assert "fq003" in report_text
    assert "fq004" in report_text

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    ids = {row["id"] for row in rows}
    assert ids == {"fq001", "fq002"}

    error_row = next(row for row in rows if row["id"] == "fq001")
    assert error_row["generated_answer"] == ""
    assert error_row["cited_document_ids"] == ""
    assert error_row["retrieval_succeeded"] == ""
