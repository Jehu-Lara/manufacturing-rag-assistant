from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from src.domain.models import Citation, RetrievalResult

logger = logging.getLogger(__name__)

RRF_K = 60


def rrf_scores(ranked_chunk_ids: Sequence[str], k: int = RRF_K) -> dict[str, float]:
    return {chunk_id: 1.0 / (k + rank) for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)}


def fuse_rankings(
    semantic_ranked_ids: Sequence[str], bm25_ranked_ids: Sequence[str], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Pure RRF fusion over two already-ranked id lists. Returns (chunk_id,
    fused_score) pairs sorted by (-fused_score, chunk_id) — the tie-break is
    part of the policy, not the orchestration. No metadata, no I/O."""
    semantic_rrf = rrf_scores(semantic_ranked_ids, k)
    bm25_rrf = rrf_scores(bm25_ranked_ids, k)
    all_ids = set(semantic_ranked_ids) | set(bm25_ranked_ids)
    fused = [(cid, semantic_rrf.get(cid, 0.0) + bm25_rrf.get(cid, 0.0)) for cid in all_ids]
    fused.sort(key=lambda pair: (-pair[1], pair[0]))
    return fused


def is_confident(top1_semantic_score: Optional[float], threshold: float) -> bool:
    return top1_semantic_score is not None and top1_semantic_score >= threshold


def top1_semantic_score_from_results(results: Sequence[RetrievalResult]) -> Optional[float]:
    """Returns None instead of raising on an empty/scoreless list, because
    here that's a legitimate "nothing relevant retrieved" signal that should
    flow into a refusal, not crash the /query endpoint."""
    for result in results:
        if result.semantic_rank == 1:
            return result.semantic_score
    scored = [result.semantic_score for result in results if result.semantic_score is not None]
    if not scored:
        return None
    fallback = max(scored)
    logger.warning(
        "no result with semantic_rank == 1 among %d results; falling back to max semantic_score (%.4f)",
        len(results),
        fallback,
    )
    return fallback


class RefusalPolicy:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def top1_semantic_score(self, results: Sequence[RetrievalResult]) -> Optional[float]:
        return top1_semantic_score_from_results(results)

    def is_confident(self, results: Sequence[RetrievalResult]) -> bool:
        return is_confident(self.top1_semantic_score(results), self._threshold)


class CitationResolver:
    @staticmethod
    def resolve(llm_citations: list[dict[str, Any]], results: Sequence[RetrievalResult]) -> list[Citation]:
        results_by_chunk_id = {result.chunk_id: result for result in results}
        resolved: list[Citation] = []
        for llm_citation in llm_citations:
            chunk_id = llm_citation.get("chunk_id")
            result = results_by_chunk_id.get(chunk_id) if isinstance(chunk_id, str) else None
            if result is None:
                logger.warning(
                    "LLM cited a chunk_id not among the retrieved chunks; dropping citation",
                    extra={"event": "citation_not_in_retrieved_set", "chunk_id": chunk_id},
                )
                continue
            metadata = result.metadata
            resolved.append(
                Citation(
                    document_id=metadata["document_id"],
                    document_title=metadata["document_title"],
                    section_heading=metadata["section_heading"],
                    revision=metadata["revision"],
                    chunk_id=result.chunk_id,
                )
            )
        return resolved
