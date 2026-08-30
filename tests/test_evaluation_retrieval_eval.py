from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.features.evaluation.retrieval_eval as retrieval_eval
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
                {"id": "q1", "answerable": True},
                {"id": "u1", "answerable": False},
            ],
        },
    )
    monkeypatch.setattr(retrieval_eval, "build_retriever", lambda mode: object())
    monkeypatch.setattr(
        retrieval_eval, "_score_answerable", lambda r, q: _answerable_row("q1", "en", 0.8)
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
