from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from retrieval import bm25_index, vector_store

RRF_K = 60
DEFAULT_TOP_N = 20


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    fused_score: float
    semantic_rank: Optional[int]
    semantic_score: Optional[float]
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    metadata: dict


def _rrf_scores(ranked_ids: list[str], k: int = RRF_K) -> dict[str, float]:
    return {chunk_id: 1.0 / (k + rank) for rank, chunk_id in enumerate(ranked_ids, start=1)}


def _sort_fused_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    results.sort(key=lambda result: (-result.fused_score, result.chunk_id))
    return results


def retrieve(query_text: str, k: int = 5, top_n: int = DEFAULT_TOP_N) -> list[RetrievalResult]:
    semantic_hits = vector_store.query(query_text, top_n)
    bm25_hits = bm25_index.query(query_text, top_n)

    semantic_by_id = {
        chunk_id: (rank, score, metadata)
        for rank, (chunk_id, score, metadata) in enumerate(semantic_hits, start=1)
    }
    bm25_by_id = {chunk_id: (rank, score) for rank, (chunk_id, score) in enumerate(bm25_hits, start=1)}

    semantic_rrf = _rrf_scores(list(semantic_by_id.keys()))
    bm25_rrf = _rrf_scores(list(bm25_by_id.keys()))

    fused: list[RetrievalResult] = []
    for chunk_id in set(semantic_by_id) | set(bm25_by_id):
        sem = semantic_by_id.get(chunk_id)
        bm = bm25_by_id.get(chunk_id)
        metadata = sem[2] if sem else vector_store.get_metadata(chunk_id)
        fused.append(
            RetrievalResult(
                chunk_id=chunk_id,
                fused_score=semantic_rrf.get(chunk_id, 0.0) + bm25_rrf.get(chunk_id, 0.0),
                semantic_rank=sem[0] if sem else None,
                semantic_score=sem[1] if sem else None,
                bm25_rank=bm[0] if bm else None,
                bm25_score=bm[1] if bm else None,
                metadata=metadata,
            )
        )

    fused = _sort_fused_results(fused)
    return fused[:k]
