from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import src.features.evaluation.retrieval_eval as retrieval_eval
from src.domain.models import RetrievalResult
from src.features.evaluation.retrieval_eval import build_report


def _answerable_row(row_id: str, language: str, top1_semantic: float) -> dict[str, Any]:
    return {
        "id": row_id,
        "language": language,
        "retrieved": [],
        "recall@3": 1.0,
        "recall@5": 1.0,
        "rr": 1.0,
        "top1_semantic_score": top1_semantic,
    }


def _unanswerable_row(row_id: str, fused: float, semantic: float) -> dict[str, Any]:
    return {
        "id": row_id,
        "retrieved": [],
        "top1_fused_score": fused,
        "top1_semantic_score": semantic,
    }


def _question(row_id: str, language: str, expected_chunk_ids: list[str]) -> dict[str, Any]:
    return {
        "id": row_id,
        "language": language,
        "question": f"question {row_id}",
        "answerable": bool(expected_chunk_ids),
        "expected_chunk_ids": expected_chunk_ids,
    }


def _fixtures() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    questions = [
        _question("q_en_1", "en", ["A"]),
        _question("q_es_1", "es", ["A"]),
        _question("q_en_2", "en", ["B", "C"]),
        _question("q_es_2", "es", ["C", "B"]),
        _question("q_en_3", "en", ["D"]),
        _question("u1", "en", []),
        _question("u2", "en", []),
    ]
    answerable_rows = [
        _answerable_row("q_en_1", "en", 0.80),
        _answerable_row("q_es_1", "es", 0.60),
        _answerable_row("q_en_2", "en", 0.70),
        _answerable_row("q_es_2", "es", 0.75),
        _answerable_row("q_en_3", "en", 0.66),
    ]
    unanswerable_rows = [
        _unanswerable_row("u1", 0.10, 0.20),
        _unanswerable_row("u2", 0.15, 0.25),
    ]
    return questions, answerable_rows, unanswerable_rows


def test_build_report_has_matched_pair_gap_section_with_per_pair_values_and_mean():
    questions, answerable_rows, unanswerable_rows = _fixtures()

    report = build_report(questions, answerable_rows, unanswerable_rows, "9.9.9")

    assert "## Matched-pair cosine gap (en − es)" in report
    # q_en_1/q_es_1 share {"A"}; q_en_2/q_es_2 share {"B","C"} (set-equal).
    assert "| q_en_1 | q_es_1 | 0.8000 | 0.6000 | +0.2000 |" in report
    assert "| q_en_2 | q_es_2 | 0.7000 | 0.7500 | -0.0500 |" in report
    # mean of (+0.20, -0.05) over the two matched pairs
    assert "**Mean gap (en − es), n=2**: +0.0750" in report


def test_build_report_matched_pair_section_ignores_unpaired_english_question():
    questions, answerable_rows, unanswerable_rows = _fixtures()

    report = build_report(questions, answerable_rows, unanswerable_rows, "9.9.9")

    # q_en_3 (expected {"D"}) has no Spanish counterpart -> no matched-pair row.
    assert "| q_en_3 | q_es" not in report
    assert "**Mean gap (en − es), n=2**" in report


def test_build_report_matched_pair_section_reports_none_when_no_pairs():
    questions = [
        _question("q_en_1", "en", ["A"]),
        _question("q_es_1", "es", ["Z"]),
        _question("u1", "en", []),
    ]
    answerable_rows = [
        _answerable_row("q_en_1", "en", 0.80),
        _answerable_row("q_es_1", "es", 0.60),
    ]
    unanswerable_rows = [_unanswerable_row("u1", 0.10, 0.20)]

    report = build_report(questions, answerable_rows, unanswerable_rows, "9.9.9")

    assert "_No en/es answerable pairs share `expected_chunk_ids`._" in report


class _FakeHeader:
    def render(self) -> str:
        return "## Provenance\n- index_profile: fake\n"


def _patch_run(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(retrieval_eval, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(retrieval_eval.eval_set_integrity, "verify", lambda: None)
    monkeypatch.setattr(
        retrieval_eval.eval_set_integrity,
        "load_eval_set",
        lambda: {
            "version": "9.9.9",
            "questions": [
                {
                    "id": "q1",
                    "language": "en",
                    "answerable": True,
                    "expected_document_id": "doc-a",
                    "expected_chunk_ids": ["doc-a::chunk-0001"],
                },
                {"id": "u1", "answerable": False},
            ],
        },
    )
    monkeypatch.setattr(retrieval_eval, "build_retriever", lambda mode: object())
    monkeypatch.setattr(retrieval_eval, "assert_live_index_profile", lambda p: None)
    monkeypatch.setattr(
        retrieval_eval,
        "_score_answerable",
        lambda r, q: {**_answerable_row("q1", "en", 0.8), "gate_confident": True, "retrieved": []},
    )
    monkeypatch.setattr(
        retrieval_eval, "_score_unanswerable", lambda r, q: _unanswerable_row("u1", 0.1, 0.2)
    )
    monkeypatch.setattr(retrieval_eval, "build_report", lambda *a, **k: "# stub\n")
    monkeypatch.setattr(
        retrieval_eval.artifacts, "resolve_provenance", lambda ip, em: _FakeHeader()
    )


def test_run_raw_off_writes_profile_and_mode_suffixed_report(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)

    path = retrieval_eval.run("off")

    assert path == tmp_path / "retrieval_report_v9.9.9__raw-v1__off.md"
    assert "v9.9.9" in path.name
    assert path.read_text(encoding="utf-8").endswith("# stub\n")


def test_run_off_and_semantic_produce_distinct_paths(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)

    off_path = retrieval_eval.run("off", index_profile="raw-v1")
    semantic_path = retrieval_eval.run("semantic", index_profile="raw-v1")

    assert off_path != semantic_path
    assert off_path == tmp_path / "retrieval_report_v9.9.9__raw-v1__off.md"
    assert semantic_path == tmp_path / "retrieval_report_v9.9.9__raw-v1__semantic.md"
    assert "v9.9.9" in off_path.name and "v9.9.9" in semantic_path.name


def test_run_canonical_alias_rejected_for_non_baseline(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        retrieval_eval.run("semantic", index_profile="raw-v1", write_canonical_alias=True)
    with pytest.raises(ValueError):
        retrieval_eval.run("off", index_profile="contextual-v1", write_canonical_alias=True)


def test_run_canonical_alias_written_only_for_raw_off(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)

    path = retrieval_eval.run("off", index_profile="raw-v1", write_canonical_alias=True)

    assert path == tmp_path / "retrieval_report_v9.9.9__raw-v1__off.md"
    assert (tmp_path / "retrieval_report_v9.9.9.md").exists()


def test_run_emits_suffixed_retrieval_details_jsonl(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)

    retrieval_eval.run("off", index_profile="raw-v1")

    details = tmp_path / "retrieval_details_v9.9.9__raw-v1__off.jsonl"
    assert details.exists()
    assert not (tmp_path / (details.name + ".tmp")).exists()
    obj = json.loads(details.read_text(encoding="utf-8").splitlines()[0])
    assert obj["id"] == "q1"
    assert obj["lang"] == "en"
    assert obj["expected_document_id"] == "doc-a"


def _rr(
    chunk_id: str,
    fused: float,
    sem_rank: int | None,
    sem_score: float | None,
    bm25_rank: int | None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=fused,
        semantic_rank=sem_rank,
        semantic_score=sem_score,
        bm25_rank=bm25_rank,
        bm25_score=None if bm25_rank is None else 1.0,
        metadata={"chunk_id": chunk_id},
    )


def test_retrieval_detail_record_shape_and_nulls() -> None:
    question = {
        "id": "q001",
        "language": "en",
        "expected_document_id": "doc-a",
        "expected_chunk_ids": ["doc-a::chunk-0003"],
    }
    row = {
        "id": "q001",
        "gate_confident": True,
        "retrieved": [
            _rr("doc-a::chunk-0001", 0.03, 1, 0.71, 2),
            _rr("doc-b::chunk-0009", 0.016, None, None, 1),
            _rr("doc-a::chunk-0002", 0.015, 3, 0.55, None),
            _rr("doc-a::chunk-0003", 0.014, 4, 0.50, None),
            _rr("doc-c::chunk-0000", 0.013, 5, 0.48, None),
            _rr("doc-x::chunk-0000", 0.001, 9, 0.10, None),
        ],
    }

    record = retrieval_eval._retrieval_detail_record(question, row)

    assert record["id"] == "q001"
    assert record["lang"] == "en"
    assert record["gate_decision"] == "answer"
    assert record["expected_document_id"] == "doc-a"
    assert record["expected_chunk_ids"] == ["doc-a::chunk-0003"]
    # top5 is exactly 5 even though 6 were retrieved, in fused-rank order
    assert [e["chunk_id"] for e in record["top5"]] == [
        "doc-a::chunk-0001",
        "doc-b::chunk-0009",
        "doc-a::chunk-0002",
        "doc-a::chunk-0003",
        "doc-c::chunk-0000",
    ]
    assert [e["rank"] for e in record["top5"]] == [1, 2, 3, 4, 5]
    # absent semantic rank/score -> None, never 0
    absent = record["top5"][1]
    assert absent["semantic_rank"] is None and absent["semantic_score"] is None
    assert record["top5"][2]["bm25_rank"] is None
    dumped = json.dumps(record)
    assert '"semantic_score": null' in dumped
    assert '"bm25_rank": null' in dumped


def test_write_retrieval_details_is_atomic_and_newline_terminated(tmp_path: Path) -> None:
    question = {
        "id": "q001",
        "language": "es",
        "expected_document_id": "d",
        "expected_chunk_ids": ["d::chunk-1"],
    }
    row = {"id": "q001", "gate_confident": False, "retrieved": [_rr("d::chunk-1", 0.02, 1, 0.4, 1)]}
    out = tmp_path / "retrieval_details_v1.1.0__raw-v1__off.jsonl"

    retrieval_eval._write_retrieval_details(out, [question], [row])

    assert out.exists()
    assert not (tmp_path / (out.name + ".tmp")).exists()
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text.splitlines()[0])["gate_decision"] == "refuse"
