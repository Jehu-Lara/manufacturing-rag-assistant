from __future__ import annotations

import pytest

from src.domain.models import RetrievalResult
from src.features.evaluation import regression_eval
from src.features.evaluation.regression_eval import _row, run


class _FakeHeader:
    def render(self) -> str:
        return "## Provenance\n- index_profile: fake\n"

THRESHOLD = 0.5999


def _result(chunk_id: str, semantic_score: float, semantic_rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=1.0,
        semantic_rank=semantic_rank,
        semantic_score=semantic_score,
        bm25_rank=1,
        bm25_score=1.0,
        metadata={"chunk_id": chunk_id},
    )


class _ScriptedRetriever:
    """RetrieverPort fake — returns a fixed result list for every query so no
    embedding model or vector index is loaded."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        return list(self._results)


def test_row_gate_answer_when_top1_semantic_at_or_above_threshold():
    retriever = _ScriptedRetriever([_result("doc::chunk-0007", 0.83)])
    query = {
        "id": "r001",
        "query": "What is the difference between NPSHA and NPSHR?",
        "language": "en",
        "expected_chunk_id": "doc::chunk-0007",
        "should_answer": True,
    }

    row = _row(retriever, query, THRESHOLD)

    assert row["gate"] == "answer"
    assert row["recall@5"] == 1.0
    assert row["top1_semantic"] == 0.83
    assert row["top1_chunk"] == "doc::chunk-0007"


def test_row_gate_refuse_when_top1_semantic_below_threshold():
    retriever = _ScriptedRetriever([_result("doc::chunk-0001", 0.41)])
    query = {
        "id": "r002",
        "query": "algo",
        "language": "es",
        "expected_chunk_id": "doc::chunk-0007",
        "should_answer": True,
    }

    row = _row(retriever, query, THRESHOLD)

    assert row["gate"] == "REFUSE"
    assert row["recall@5"] == 0.0


def test_row_recall_is_none_for_control_queries_without_expected_chunk():
    retriever = _ScriptedRetriever([_result("doc::chunk-0009", 0.7)])
    query = {
        "id": "r018",
        "query": "What NPSH margin does API 610 recommend?",
        "language": "en",
        "expected_chunk_id": None,
        "should_answer": False,
    }

    row = _row(retriever, query, THRESHOLD)

    assert row["recall@5"] is None
    assert row["gate"] == "answer"


def test_run_writes_report_with_expansion_mode_heading(tmp_path, monkeypatch):
    retriever = _ScriptedRetriever([_result("doc::chunk-0007", 0.7)])
    monkeypatch.setattr(regression_eval, "build_retriever", lambda mode: retriever)
    monkeypatch.setattr(regression_eval, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(regression_eval.artifacts, "resolve_provenance", lambda ip, em: _FakeHeader())

    report_path = run("off")

    assert report_path.exists()
    assert report_path.parent == tmp_path
    assert report_path.name == "regression_eval_v1.1.0__raw-v1__off.md"
    text = report_path.read_text(encoding="utf-8")
    assert "## expansion_mode = off" in text
    assert "| id | lang | should_answer | top1_semantic | gate | recall@5 |" in text
    assert "answerable passing gate:" in text
    assert "controls correctly refused:" in text


def test_run_off_and_semantic_produce_distinct_paths(tmp_path, monkeypatch):
    retriever = _ScriptedRetriever([_result("doc::chunk-0007", 0.7)])
    monkeypatch.setattr(regression_eval, "build_retriever", lambda mode: retriever)
    monkeypatch.setattr(regression_eval, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(regression_eval.artifacts, "resolve_provenance", lambda ip, em: _FakeHeader())

    off_path = run("off", index_profile="raw-v1")
    semantic_path = run("semantic", index_profile="raw-v1")

    assert off_path != semantic_path
    assert off_path.name == "regression_eval_v1.1.0__raw-v1__off.md"
    assert semantic_path.name == "regression_eval_v1.1.0__raw-v1__semantic.md"
    assert "v1.1.0" in off_path.name and "v1.1.0" in semantic_path.name


def test_run_canonical_alias_guarded(tmp_path, monkeypatch):
    retriever = _ScriptedRetriever([_result("doc::chunk-0007", 0.7)])
    monkeypatch.setattr(regression_eval, "build_retriever", lambda mode: retriever)
    monkeypatch.setattr(regression_eval, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(regression_eval.artifacts, "resolve_provenance", lambda ip, em: _FakeHeader())

    with pytest.raises(ValueError):
        run("semantic", index_profile="raw-v1", write_canonical_alias=True)

    path = run("off", index_profile="raw-v1", write_canonical_alias=True)
    assert path.name == "regression_eval_v1.1.0__raw-v1__off.md"
    assert (tmp_path / "regression_eval_v1.1.0.md").exists()
