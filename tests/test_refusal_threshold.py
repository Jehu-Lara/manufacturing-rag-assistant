from __future__ import annotations

import logging
from dataclasses import dataclass

from api.config import load_settings
from api.refusal import (
    is_confident,
    is_confident_for_results,
    top1_semantic_score_from_results,
)


@dataclass
class _FakeResult:
    """Duck-typed stand-in for retrieval.hybrid.RetrievalResult — only the two
    attributes top1_semantic_score_from_results actually reads."""

    semantic_rank: int | None
    semantic_score: float | None


# Boundary convention confirmed from eval/threshold_analysis.py's
# sweep_thresholds()/select_threshold() logic: `_refusal_counts` computes
# `answerable_wrongly_refused = sum(1 for score in answerable_scores if score < threshold)`.
# A score exactly equal to the threshold is NOT counted as refused there, i.e.
# score == threshold is treated as "would be answered" (confident). `is_confident`
# below matches this: `>= threshold` is confident (not refused).


def test_is_confident_score_above_threshold_is_true():
    assert is_confident(0.7, 0.5599) is True


def test_is_confident_score_below_threshold_is_false():
    assert is_confident(0.3, 0.5599) is False


def test_is_confident_score_exactly_at_threshold_is_true():
    assert is_confident(0.5599, 0.5599) is True


def test_is_confident_none_score_is_false_no_exception():
    assert is_confident(None, 0.5599) is False


def test_top1_semantic_score_from_results_returns_rank_one_not_results_zero():
    results = [
        _FakeResult(semantic_rank=3, semantic_score=0.99),
        _FakeResult(semantic_rank=1, semantic_score=0.42),
        _FakeResult(semantic_rank=2, semantic_score=0.77),
    ]
    assert top1_semantic_score_from_results(results) == 0.42


def test_top1_semantic_score_from_results_falls_back_to_max_and_logs_warning(caplog):
    results = [
        _FakeResult(semantic_rank=2, semantic_score=0.55),
        _FakeResult(semantic_rank=3, semantic_score=0.61),
    ]
    with caplog.at_level(logging.WARNING, logger="api.refusal"):
        score = top1_semantic_score_from_results(results)
    assert score == 0.61
    assert any("semantic_rank == 1" in record.message for record in caplog.records)


def test_top1_semantic_score_from_results_empty_list_returns_none_no_exception():
    assert top1_semantic_score_from_results([]) is None


def test_top1_semantic_score_from_results_all_none_scores_returns_none_no_exception():
    results = [
        _FakeResult(semantic_rank=None, semantic_score=None),
        _FakeResult(semantic_rank=None, semantic_score=None),
    ]
    assert top1_semantic_score_from_results(results) is None


def test_is_confident_for_results_end_to_end_true_case():
    results = [
        _FakeResult(semantic_rank=2, semantic_score=0.30),
        _FakeResult(semantic_rank=1, semantic_score=0.90),
    ]
    assert is_confident_for_results(results, 0.5599) is True


def test_is_confident_for_results_end_to_end_false_case():
    results = [
        _FakeResult(semantic_rank=1, semantic_score=0.10),
    ]
    assert is_confident_for_results(results, 0.5599) is False


def test_settings_default_refusal_threshold_loads_and_gates_correctly(monkeypatch):
    monkeypatch.delenv("REFUSAL_COSINE_THRESHOLD", raising=False)
    settings = load_settings()
    assert settings.refusal_cosine_threshold == 0.5999
    assert is_confident(0.6, settings.refusal_cosine_threshold) is True
    assert is_confident(0.5, settings.refusal_cosine_threshold) is False
