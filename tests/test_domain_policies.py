from __future__ import annotations

import logging

from src.domain.models import Citation, RetrievalResult
from src.domain.policies import (
    GLOSSARY,
    RRF_K,
    CitationResolver,
    RefusalPolicy,
    expand_query,
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


def test_rrf_fusion_favors_items_ranked_highly_in_both_lists():
    # Hand-computed: "shared" ranks #1 semantic + #2 bm25 = 1/61 + 1/62.
    # "semantic_only" ranks #1 semantic only = 1/61. Fused score must put
    # "shared" ahead despite neither list ranking it #1 alone in both.
    fused = dict(fuse_rankings(["semantic_only", "shared"], ["other", "shared"]))
    assert fused["shared"] > fused["semantic_only"]
    assert fused["shared"] > fused["other"]


def test_refusal_policy_boundary_at_exactly_0_5999():
    threshold = 0.5999
    results = [_result("chunk-1", 1, threshold)]
    assert RefusalPolicy(threshold).is_confident(results) is True
    assert is_confident(threshold, threshold) is True
    assert is_confident(threshold - 0.0001, threshold) is False


def test_is_confident_score_above_threshold_is_true():
    assert is_confident(0.7, 0.5599) is True


def test_is_confident_score_below_threshold_is_false():
    assert is_confident(0.3, 0.5599) is False


def test_is_confident_none_score_is_false_no_exception():
    assert is_confident(None, 0.5599) is False


def test_top1_semantic_score_from_results_prefers_rank_one():
    results = [_result("chunk-1", 3, 0.99), _result("chunk-2", 1, 0.42)]
    assert top1_semantic_score_from_results(results) == 0.42


def test_top1_semantic_score_from_results_falls_back_to_max_and_logs_warning(caplog):
    results = [_result("chunk-1", 2, 0.55), _result("chunk-2", 3, 0.61)]
    with caplog.at_level(logging.WARNING, logger="src.domain.policies"):
        score = top1_semantic_score_from_results(results)
    assert score == 0.61
    assert any("semantic_rank == 1" in record.message for record in caplog.records)


def test_top1_semantic_score_from_results_empty_list_returns_none_no_exception():
    assert top1_semantic_score_from_results([]) is None


def test_top1_semantic_score_from_results_all_none_scores_returns_none_no_exception():
    results = [_result("chunk-1", None, None), _result("chunk-2", None, None)]
    assert top1_semantic_score_from_results(results) is None


def test_refusal_policy_end_to_end_true_case():
    results = [_result("chunk-1", 2, 0.30), _result("chunk-2", 1, 0.90)]
    assert RefusalPolicy(0.5599).is_confident(results) is True


def test_refusal_policy_end_to_end_false_case():
    results = [_result("chunk-1", 1, 0.10)]
    assert RefusalPolicy(0.5599).is_confident(results) is False


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


def test_citation_resolver_empty_llm_citations_returns_empty_list_no_error():
    results = [_result("chunk-1", 1, 0.9)]

    resolved = CitationResolver.resolve([], results)

    assert resolved == []


def test_expand_query_passthrough_when_no_glossary_key():
    q = "What is the frame level tolerance for the conveyor?"
    assert expand_query(q) == q


def test_expand_query_appends_expansions_for_known_acronym():
    out = expand_query("What is the difference between NPSHA and NPSHR?")
    assert out.startswith("What is the difference between NPSHA and NPSHR?")
    assert "net positive suction head available" in out
    assert "net positive suction head required" in out
    assert "altura neta de succión positiva disponible" in out


def test_expand_query_is_case_insensitive_and_word_bounded():
    assert "net positive suction head available" in expand_query("what is npsha")
    unchanged = "the NPSHATEST rig and xNPSHA probe"
    assert expand_query(unchanged) == unchanged


def test_expand_query_multi_term_order_is_deterministic():
    a = expand_query("PEL and IDLH for acetone")
    b = expand_query("IDLH and PEL for acetone")
    # additions ordered by GLOSSARY insertion order, not query order → identical tails
    assert a.split("for acetone", 1)[1] == b.split("for acetone", 1)[1]
    assert expand_query("PEL and IDLH for acetone") == a  # stable across calls


def test_expand_query_dedupes_expansion_already_present():
    q = "define permissible exposure limit PEL"
    out = expand_query(q)
    assert out.count("permissible exposure limit") == 1


def test_glossary_english_expansions_are_corpus_attested():
    import pathlib

    corpus_text = " ".join(
        p.read_text(encoding="utf-8").lower()
        for p in pathlib.Path("corpus").rglob("*.md")
    )
    for key, expansions in GLOSSARY.items():
        english = expansions[0]
        assert english.lower() in corpus_text, f"{key}: {english!r} not found in corpus"


def test_glossary_spanish_renderings_nonempty_and_distinct():
    for key, expansions in GLOSSARY.items():
        assert len(expansions) >= 2, f"{key}: needs an es rendering"
        es = expansions[1]
        assert es.strip()
        assert es.lower() != expansions[0].lower()
