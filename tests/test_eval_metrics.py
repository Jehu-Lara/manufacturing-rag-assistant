from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.metrics import mean_reciprocal_rank, reciprocal_rank, recall_at_k, top1_semantic_score


@dataclass
class _FakeResult:
    """Duck-typed stand-in for retrieval.hybrid.RetrievalResult — only the two
    attributes top1_semantic_score actually reads."""

    semantic_rank: int | None
    semantic_score: float | None


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a", "b", "c", "d"], ["d"], k=3) == 0.0


def test_recall_at_k_partial_hit_with_multiple_expected():
    assert recall_at_k(["a", "x", "y"], ["a", "z"], k=3) == 0.5


def test_recall_at_k_rejects_empty_expected():
    with pytest.raises(ValueError):
        recall_at_k(["a", "b"], [], k=3)


def test_reciprocal_rank_hit_at_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_hit_at_third_position():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit_returns_zero():
    assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_mean_reciprocal_rank_averages_correctly():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_mean_reciprocal_rank_rejects_empty_input():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])


def test_top1_semantic_score_returns_semantic_rank_one_not_results_zero():
    # results[0] (fused-order top-1) deliberately has a different, higher score
    # than the semantic_rank==1 item, to prove the function doesn't just read
    # results[0] — that's exactly the fused-vs-semantic mixup this function
    # exists to avoid.
    results = [
        _FakeResult(semantic_rank=3, semantic_score=0.99),
        _FakeResult(semantic_rank=1, semantic_score=0.42),
        _FakeResult(semantic_rank=2, semantic_score=0.77),
    ]
    assert top1_semantic_score(results) == 0.42


def test_top1_semantic_score_falls_back_to_max_when_no_rank_one_present():
    results = [
        _FakeResult(semantic_rank=2, semantic_score=0.55),
        _FakeResult(semantic_rank=3, semantic_score=0.61),
    ]
    assert top1_semantic_score(results) == 0.61


def test_top1_semantic_score_logs_warning_to_stderr_on_fallback(capsys):
    results = [
        _FakeResult(semantic_rank=2, semantic_score=0.55),
        _FakeResult(semantic_rank=3, semantic_score=0.61),
    ]
    top1_semantic_score(results)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "WARNING" in captured.err
    assert "semantic_rank == 1" in captured.err
    assert "0.61" in captured.err


def test_top1_semantic_score_does_not_warn_when_rank_one_present(capsys):
    results = [_FakeResult(semantic_rank=1, semantic_score=0.5)]
    top1_semantic_score(results)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_top1_semantic_score_rejects_empty_results():
    with pytest.raises(ValueError):
        top1_semantic_score([])


def test_top1_semantic_score_raises_when_no_result_has_a_semantic_score():
    results = [
        _FakeResult(semantic_rank=None, semantic_score=None),
        _FakeResult(semantic_rank=None, semantic_score=None),
    ]
    with pytest.raises(ValueError):
        top1_semantic_score(results)
