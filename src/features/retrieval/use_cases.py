from __future__ import annotations

from typing import Optional

from src.core.telemetry import get_tracer
from src.domain.models import ExpansionMode, RetrievalResult
from src.domain.policies import expand_query, fuse_rankings
from src.domain.ports import LexicalIndexPort, RerankerPort, VectorStorePort

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

    def __init__(
        self,
        vector_store: VectorStorePort,
        lexical_index: LexicalIndexPort,
        expansion_mode: ExpansionMode = "off",
        *,
        reranker: Optional[RerankerPort] = None,
        rerank_window: int = DEFAULT_TOP_N,
    ) -> None:
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._expansion_mode = expansion_mode
        self._reranker = reranker
        self._rerank_window = rerank_window

    def retrieve(self, query_text: str, k: int = 5, top_n: int = DEFAULT_TOP_N) -> list[RetrievalResult]:
        with get_tracer().start_as_current_span("retrieval.hybrid.query"):
            # Production runs "off", where neither channel can use the
            # expansion — computing it anyway meant every served query paid for
            # a glossary scan whose result was then discarded.
            if self._expansion_mode == "off":
                semantic_query = lexical_query = query_text
            else:
                expanded = expand_query(query_text)
                semantic_query = expanded if self._expansion_mode in ("semantic", "both") else query_text
                lexical_query = expanded if self._expansion_mode in ("lexical", "both") else query_text
            semantic_hits = self._vector_store.query(semantic_query, top_n)
            bm25_hits = self._lexical_index.query(lexical_query, top_n)

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

            if self._reranker is not None:
                fused = self._apply_reranker(query_text, fused)

            return fused[:k]

    def _apply_reranker(self, query_text: str, fused: list[RetrievalResult]) -> list[RetrievalResult]:
        """Permutes ONLY the first rerank_window entries and leaves the tail in
        fused order. Nothing is added, dropped or deduplicated, so the result
        with semantic_rank == 1 is still present and the refusal gate reads the
        same score it would without a reranker (see
        src.domain.policies.top1_semantic_score_from_results). A reranker that
        truncated before the gate read it would silently retune 0.5999/0.5500.

        Score fields are carried through untouched: reranking reorders objects,
        it never rewrites what a channel measured."""
        assert self._reranker is not None
        window, tail = fused[: self._rerank_window], fused[self._rerank_window :]
        if len(window) < 2:
            return fused
        by_id = {result.chunk_id: result for result in window}
        scored = self._reranker.rerank(
            query_text, [(r.chunk_id, str(r.metadata.get("chunk_text", ""))) for r in window]
        )
        reranked_ids = [chunk_id for chunk_id, _ in scored]
        if len(reranked_ids) != len(window) or set(reranked_ids) != set(by_id):
            raise ValueError("reranker must return the same id set it was given")
        return [by_id[chunk_id] for chunk_id, _ in scored] + tail
