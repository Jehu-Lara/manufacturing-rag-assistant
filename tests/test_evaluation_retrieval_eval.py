from __future__ import annotations

from typing import Any

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
