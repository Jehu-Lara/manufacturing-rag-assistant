from __future__ import annotations

from retrieval.hybrid import RetrievalResult
from retrieval.hybrid import _rrf_scores
from retrieval.hybrid import _sort_fused_results
from retrieval.hybrid import retrieve


def _sorted_chunk_ids(results: list[RetrievalResult]) -> list[str]:
    return [result.chunk_id for result in _sort_fused_results(results)]


def test_rrf_scores_match_hand_computed_values():
    scores = _rrf_scores(["a", "b", "c"], k=60)
    assert scores == {
        "a": 1.0 / 61,
        "b": 1.0 / 62,
        "c": 1.0 / 63,
    }


def test_rrf_fusion_favors_items_ranked_highly_in_both_lists():
    # Hand-computed: "shared" ranks #1 semantic + #2 bm25 = 1/61 + 1/62.
    # "semantic_only" ranks #1 semantic only = 1/61. Fused score must put
    # "shared" ahead despite neither list ranking it #1 alone in both.
    semantic_rrf = _rrf_scores(["semantic_only", "shared"], k=60)
    bm25_rrf = _rrf_scores(["other", "shared"], k=60)

    fused = {
        chunk_id: semantic_rrf.get(chunk_id, 0.0) + bm25_rrf.get(chunk_id, 0.0)
        for chunk_id in set(semantic_rrf) | set(bm25_rrf)
    }
    assert fused["shared"] > fused["semantic_only"]
    assert fused["shared"] > fused["other"]


def test_fused_score_ties_break_deterministically_on_chunk_id():
    def build_result(chunk_id: str) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            fused_score=0.05,
            semantic_rank=1,
            semantic_score=0.9,
            bm25_rank=1,
            bm25_score=10.0,
            metadata={},
        )

    tied_chunk_ids = ["zzz-chunk", "aaa-chunk", "mmm-chunk"]

    results_in_original_order = [build_result(chunk_id) for chunk_id in tied_chunk_ids]
    results_in_reversed_order = [build_result(chunk_id) for chunk_id in reversed(tied_chunk_ids)]

    expected_order = ["aaa-chunk", "mmm-chunk", "zzz-chunk"]
    assert _sorted_chunk_ids(results_in_original_order) == expected_order
    assert _sorted_chunk_ids(results_in_reversed_order) == expected_order


def test_known_query_returns_known_relevant_chunk_in_top_k():
    # Integration test against the real built index — requires
    # `python -m retrieval.build_index` to have been run first.
    results = retrieve("What is lockout/tagout and why does it matter?", k=3)
    retrieved_ids = [r.chunk_id for r in results]
    assert any(chunk_id.startswith("osha-3120-lockout-tagout::") for chunk_id in retrieved_ids), (
        f"expected a lockout/tagout chunk in top-3, got {retrieved_ids}"
    )
