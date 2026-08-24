from __future__ import annotations

import pytest

from eval.metrics import mean_reciprocal_rank, reciprocal_rank, recall_at_k


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
