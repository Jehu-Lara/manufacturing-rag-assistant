from __future__ import annotations

import logging

from src.domain.models import Citation, RetrievalResult
from src.domain.policies import (
    RRF_K,
    CitationResolver,
    RefusalPolicy,
    fuse_rankings,
    is_confident,
    rrf_scores,
    top1_semantic_score_from_results,
)


def _result(chunk_id: str, semantic_rank, semantic_score, **metadata_overrides: str) -> RetrievalResult:
    metadata = {
        "document_id": f"doc-{chunk_id}",
        "document_title": f"Title for {chunk_id}",
        "section_heading": f"Section for {chunk_id}",
        "revision": "Rev A",
        "chunk_id": chunk_id,
    }
    metadata.update(metadata_overrides)
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=1.0,
        semantic_rank=semantic_rank,
        semantic_score=semantic_score,
        bm25_rank=semantic_rank,
        bm25_score=1.0,
        metadata=metadata,
    )


def test_rrf_k_is_60_byte_stable_invariant():
    assert RRF_K == 60


def test_rrf_scores_matches_formula():
    assert rrf_scores(["a", "b"], k=60) == {"a": 1.0 / 61, "b": 1.0 / 62}


def test_fuse_rankings_ties_broken_by_ascending_chunk_id():
    # "b" and "a" are tied for last (each appears in only one ranked list),
    # so both get 1/(60+2) and the tie-break must be ascending chunk_id.
    fused = fuse_rankings(semantic_ranked_ids=["z", "a"], bm25_ranked_ids=["z", "b"])
    assert fused[0] == ("z", (1.0 / 61) + (1.0 / 61))
    assert [chunk_id for chunk_id, _ in fused[1:]] == ["a", "b"]


def test_refusal_policy_boundary_at_exactly_0_5999():
    threshold = 0.5999
    results = [_result("chunk-1", 1, threshold)]
    assert RefusalPolicy(threshold).is_confident(results) is True
    assert is_confident(threshold, threshold) is True
    assert is_confident(threshold - 0.0001, threshold) is False


def test_top1_semantic_score_from_results_prefers_rank_one():
    results = [_result("chunk-1", 3, 0.99), _result("chunk-2", 1, 0.42)]
    assert top1_semantic_score_from_results(results) == 0.42


def test_citation_resolver_resolves_from_retrieved_metadata_only():
    results = [_result("chunk-1", 1, 0.9)]
    llm_citations = [{"chunk_id": "chunk-1"}]

    resolved = CitationResolver.resolve(llm_citations, results)

    assert resolved == [
        Citation(
            document_id="doc-chunk-1",
            document_title="Title for chunk-1",
            section_heading="Section for chunk-1",
            revision="Rev A",
            chunk_id="chunk-1",
        )
    ]


def test_citation_resolver_drops_unmatched_chunk_id_and_logs_warning(caplog):
    results = [_result("chunk-1", 1, 0.9)]
    llm_citations = [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-unknown"}]

    with caplog.at_level(logging.WARNING, logger="src.domain.policies"):
        resolved = CitationResolver.resolve(llm_citations, results)

    assert len(resolved) == 1
    assert resolved[0].chunk_id == "chunk-1"
    assert any(
        record.__dict__.get("event") == "citation_not_in_retrieved_set"
        and record.__dict__.get("chunk_id") == "chunk-unknown"
        for record in caplog.records
    )
