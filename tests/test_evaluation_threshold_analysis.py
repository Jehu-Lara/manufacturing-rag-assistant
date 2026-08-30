from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.features.evaluation import threshold_analysis
from src.features.evaluation.threshold_analysis import (
    build_report,
    select_threshold,
    sweep_thresholds,
)


class _FakeHeader:
    def render(self) -> str:
        return "## Provenance\n- index_profile: fake\n"


def _patch_run(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(threshold_analysis, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(threshold_analysis.eval_set_integrity, "verify", lambda: None)
    monkeypatch.setattr(
        threshold_analysis.eval_set_integrity,
        "load_eval_set",
        lambda: {
            "version": "9.9.9",
            "questions": [
                {"id": "q1", "question": "q1?", "language": "en", "answerable": True},
                {"id": "q2", "question": "q2?", "language": "es", "answerable": True},
                {"id": "u1", "question": "u1?", "language": "en", "answerable": False},
                {"id": "u2", "question": "u2?", "language": "es", "answerable": False},
            ],
        },
    )
    monkeypatch.setattr(threshold_analysis, "build_retriever", lambda mode="off": object())
    scores = iter([0.80, 0.75, 0.20, 0.25])
    monkeypatch.setattr(threshold_analysis, "_top1_semantic_score", lambda r, q: next(scores))
    monkeypatch.setattr(
        threshold_analysis.artifacts, "resolve_provenance", lambda ip, em: _FakeHeader()
    )


def test_run_off_and_semantic_produce_distinct_paths(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)
    off_path = threshold_analysis.run("off", index_profile="raw-v1")
    _patch_run(monkeypatch, tmp_path)
    semantic_path = threshold_analysis.run("semantic", index_profile="raw-v1")

    assert off_path == tmp_path / "threshold_analysis_v9.9.9__raw-v1__off.md"
    assert semantic_path == tmp_path / "threshold_analysis_v9.9.9__raw-v1__semantic.md"
    assert off_path != semantic_path
    assert "v9.9.9" in off_path.name and "v9.9.9" in semantic_path.name


def test_run_canonical_alias_guarded(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_run(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        threshold_analysis.run("semantic", index_profile="raw-v1", write_canonical_alias=True)

    _patch_run(monkeypatch, tmp_path)
    path = threshold_analysis.run("off", index_profile="raw-v1", write_canonical_alias=True)
    assert (tmp_path / "threshold_analysis_v9.9.9.md").exists()
    assert path == tmp_path / "threshold_analysis_v9.9.9__raw-v1__off.md"


def test_sweep_thresholds_covers_range_inclusive_of_endpoints():
    thresholds = sweep_thresholds(0.0, 0.1, step=0.02)
    assert thresholds == [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


def test_no_overlap_branch_chooses_midpoint():
    unanswerable_scores = [0.10, 0.12, 0.15]
    answerable_scores = [0.30, 0.32, 0.40]

    result = select_threshold(unanswerable_scores, answerable_scores)

    assert result["branch"] == "no_overlap"
    assert result["threshold"] == (0.15 + 0.30) / 2


def test_overlap_branch_maximizes_correct_minus_wrong_refusals():
    # max(unanswerable)=0.26 >= min(answerable)=0.20 -> overlap. Hand-computed
    # objective (unanswerable_correctly_refused - answerable_wrongly_refused)
    # over the 0.16..0.36 sweep peaks at 1, first reached at threshold=0.20
    # (refuses the 0.18 unanswerable score, wrongly refuses nothing).
    unanswerable_scores = [0.18, 0.26]
    answerable_scores = [0.20, 0.35]

    result = select_threshold(unanswerable_scores, answerable_scores, step=0.02)

    assert result["branch"] == "overlap"
    assert result["threshold"] == pytest.approx(0.20)

    chosen_row = next(row for row in result["sweep"] if row["threshold"] == pytest.approx(0.20))
    assert chosen_row["unanswerable_correctly_refused"] == 1
    assert chosen_row["answerable_wrongly_refused"] == 0
    assert chosen_row["objective"] == max(row["objective"] for row in result["sweep"])


def test_overlap_branch_tie_breaks_toward_lowest_threshold():
    # max(unanswerable)=0.30 >= min(answerable)=0.24 -> overlap. Thresholds
    # 0.20, 0.22, and 0.24 all achieve the same best objective (1): each
    # refuses the 0.18 unanswerable score and wrongly refuses nothing. The
    # rule requires picking the lowest of those tied candidates.
    unanswerable_scores = [0.18, 0.30]
    answerable_scores = [0.24]

    result = select_threshold(unanswerable_scores, answerable_scores, step=0.02)

    assert result["branch"] == "overlap"
    best_objective = max(row["objective"] for row in result["sweep"])
    tied_thresholds = [row["threshold"] for row in result["sweep"] if row["objective"] == best_objective]
    assert len(tied_thresholds) > 1, "fixture must actually produce a tie to test tie-breaking"
    assert result["threshold"] == min(tied_thresholds)
    assert result["threshold"] == pytest.approx(0.20)


def test_build_report_pooled_selection_is_unchanged_and_still_present():
    # Synthetic fake eval set (v9.9.9): pooled scores + parallel language tags.
    unanswerable_scores = [0.18, 0.26, 0.20, 0.28]
    answerable_scores = [0.30, 0.42, 0.33, 0.45]
    unanswerable_languages = ["en", "en", "es", "es"]
    answerable_languages = ["en", "en", "es", "es"]

    selection = select_threshold(unanswerable_scores, answerable_scores)
    report = build_report(
        unanswerable_scores,
        answerable_scores,
        unanswerable_languages,
        answerable_languages,
        selection,
        "9.9.9",
    )

    assert "## Selection procedure and result" in report
    assert (
        "**Analyzer-selected threshold on this eval set (diagnostic only — NOT applied; "
        f"production REFUSAL_COSINE_THRESHOLD stays 0.5999): {selection['threshold']:.4f}**"
    ) in report


def test_build_report_includes_per_language_sweep_sections():
    unanswerable_scores = [0.18, 0.26, 0.20, 0.28]
    answerable_scores = [0.30, 0.42, 0.33, 0.45]
    unanswerable_languages = ["en", "en", "es", "es"]
    answerable_languages = ["en", "en", "es", "es"]

    selection = select_threshold(unanswerable_scores, answerable_scores)
    report = build_report(
        unanswerable_scores,
        answerable_scores,
        unanswerable_languages,
        answerable_languages,
        selection,
        "9.9.9",
    )

    assert "## English — answerable/unanswerable top-1 semantic_score" in report
    assert "## English — cutoff sweep" in report
    assert "## Spanish — answerable/unanswerable top-1 semantic_score" in report
    assert "## Spanish — cutoff sweep" in report
    assert (
        "Per-language tables are diagnostic; the shipped threshold remains the "
        "pooled selection (0.5999 override, SPEC.md Phase 3)." in report
    )


def test_select_threshold_rejects_empty_score_lists():
    with pytest.raises(ValueError):
        select_threshold([], [0.5])
    with pytest.raises(ValueError):
        select_threshold([0.5], [])
