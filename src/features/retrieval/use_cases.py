from __future__ import annotations

from src.domain.models import RetrievalResult
from src.domain.policies import fuse_rankings
from src.domain.ports import LexicalIndexPort, VectorStorePort

DEFAULT_TOP_N = 20

# retrieve()'s fused list is bounded by 2 * top_n (worst case: semantic and
# BM25 hit sets don't overlap at all), so requesting this many results
# guarantees the pure-semantic top-1 item (semantic_rank == 1) is present in
# the returned list rather than having already been cut off by a small k —
# see src.features.evaluation.metrics.top1_semantic_score /
# src.domain.policies.top1_semantic_score_from_results.
SEMANTIC_EXTRACTION_K = DEFAULT_TOP_N * 2


class HybridRetriever:
    """Implements RetrieverPort. Query serving only — never imports or is
    imported by src.features.retrieval.cli (the index-build script); they
    share this feature-package directory but not a call graph."""

    def __init__(self, vector_store: VectorStorePort, lexical_index: LexicalIndexPort) -> None:
        self._vector_store = vector_store
        self._lexical_index = lexical_index

    def retrieve(self, query_text: str, k: int = 5, top_n: int = DEFAULT_TOP_N) -> list[RetrievalResult]:
        semantic_hits = self._vector_store.query(query_text, top_n)
        bm25_hits = self._lexical_index.query(query_text, top_n)

        semantic_by_id = {
            chunk_id: (rank, score, metadata)
            for rank, (chunk_id, score, metadata) in enumerate(semantic_hits, start=1)
        }
        bm25_by_id = {chunk_id: (rank, score) for rank, (chunk_id, score) in enumerate(bm25_hits, start=1)}

        fused_pairs = fuse_rankings(list(semantic_by_id.keys()), list(bm25_by_id.keys()))

        fused: list[RetrievalResult] = []
        for chunk_id, fused_score in fused_pairs:
            sem = semantic_by_id.get(chunk_id)
            bm = bm25_by_id.get(chunk_id)
            metadata = sem[2] if sem else self._vector_store.get_metadata(chunk_id)
            fused.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    fused_score=fused_score,
                    semantic_rank=sem[0] if sem else None,
                    semantic_score=sem[1] if sem else None,
                    bm25_rank=bm[0] if bm else None,
                    bm25_score=bm[1] if bm else None,
                    metadata=metadata,
                )
            )

        return fused[:k]
