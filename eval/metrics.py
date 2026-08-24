from __future__ import annotations


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
