from __future__ import annotations

import sys
from typing import Sequence

from src.domain.models import RetrievalResult


def recall_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], k: int) -> float:
    if not expected_chunk_ids:
        raise ValueError("expected_chunk_ids must be non-empty")
    top_k = set(retrieved_chunk_ids[:k])
    hits = sum(1 for chunk_id in expected_chunk_ids if chunk_id in top_k)
    return hits / len(expected_chunk_ids)


def reciprocal_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    expected = set(expected_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    if not reciprocal_ranks:
        raise ValueError("reciprocal_ranks must be non-empty")
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def top1_semantic_score(results: Sequence[RetrievalResult]) -> float:
    """Pure-semantic top-1 score, i.e. the score of the result with `semantic_rank == 1` —
    NOT `results[0]`, which is fused-score-ranked and can be a different chunk entirely.
    """
    if not results:
        raise ValueError("results must be non-empty to extract a semantic score")
    for result in results:
        if result.semantic_rank == 1:
            assert result.semantic_score is not None
            return result.semantic_score
    scored = [result.semantic_score for result in results if result.semantic_score is not None]
    if not scored:
        raise ValueError("no result in the returned list has a semantic_score")
    fallback = max(scored)
    print(
        f"WARNING: no result with semantic_rank == 1 among {len(results)} results; "
        f"falling back to max semantic_score ({fallback:.4f})",
        file=sys.stderr,
    )
    return fallback
