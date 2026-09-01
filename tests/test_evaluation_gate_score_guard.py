from __future__ import annotations

import pytest

from src.domain.models import RetrievalResult
from src.features.evaluation import gate_score_guard


def _results(top1_score: float) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            fused_score=1.0,
            semantic_rank=1,
            semantic_score=top1_score,
            bm25_rank=1,
            bm25_score=1.0,
            metadata={},
        )
    ]


class _FakeRetriever:
    def __init__(self, score: float) -> None:
        self._score = score

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        return _results(self._score)


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.5499, False),
        (0.5500, True),
        (0.5642, True),
        (0.5998, True),
        (0.5999, False),
        (0.7000, False),
        (None, False),
    ],
)
def test_score_in_review_band_boundaries(score, expected):
    assert gate_score_guard.score_in_review_band(score) is expected


def test_run_passes_when_both_ids_in_band():
    gate_score_guard.run(retriever=_FakeRetriever(0.5642))


def test_run_fails_when_a_target_is_below_floor():
    with pytest.raises(SystemExit):
        gate_score_guard.run(retriever=_FakeRetriever(0.5100))


def test_run_fails_when_a_target_is_confident():
    with pytest.raises(SystemExit):
        gate_score_guard.run(retriever=_FakeRetriever(0.7000))
