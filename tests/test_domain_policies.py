from __future__ import annotations

import logging

import pytest

from src.domain.models import Citation, RetrievalResult
from src.domain.policies import (
    GLOSSARY,
    RRF_K,
    CitationResolver,
    GroundedEvidenceResolver,
    RefusalPolicy,
    expand_query,
    fuse_rankings,
    is_confident,
    normalize_evidence_text,
    rrf_scores,
    top1_semantic_score_from_results,
)


def _result(chunk_id: str, semantic_rank, semantic_score, **metadata_overrides: str) -> RetrievalResult:
    metadata = {
        "document_id": f"doc-{chunk_id}",
        "document_title": f"Title for {chunk_id}",
        "section_heading": f"Section for {chunk_id}",
        "revision": "Rev A",
        "source_type": "public",
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

    resolution = CitationResolver.resolve(llm_citations, results)

    assert resolution.failure_reason is None
    assert resolution.citations == [
        Citation(
            document_id="doc-chunk-1",
            document_title="Title for chunk-1",
            section_heading="Section for chunk-1",
            revision="Rev A",
            chunk_id="chunk-1",
            source_type="public",
        )
    ]


def test_citation_resolver_carries_source_type_from_retrieved_metadata():
    results = [_result("chunk-1", 1, 0.9, source_type="synthetic")]

    resolution = CitationResolver.resolve([{"chunk_id": "chunk-1"}], results)

    assert resolution.failure_reason is None
    assert resolution.citations[0].source_type == "synthetic"


def test_citation_resolver_rejects_whole_set_when_one_chunk_id_unmatched(caplog):
    """Fail-closed: a mixed valid/invalid set used to resolve partially, which
    served a partly-unverifiable answer as if it were fully cited."""
    results = [_result("chunk-1", 1, 0.9)]
    llm_citations = [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-unknown"}]

    with caplog.at_level(logging.WARNING, logger="src.domain.policies"):
        resolution = CitationResolver.resolve(llm_citations, results)

    assert resolution.citations == []
    assert resolution.failure_reason == "unresolved_citation"
    assert any(
        record.__dict__.get("event") == "citation_not_in_retrieved_set"
        and record.__dict__.get("chunk_id") == "chunk-unknown"
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "missing_field",
    ["document_id", "document_title", "section_heading", "revision", "source_type"],
)
def test_citation_resolver_rejects_when_retrieved_metadata_is_incomplete(missing_field):
    result = _result("chunk-1", 1, 0.9)
    del result.metadata[missing_field]

    resolution = CitationResolver.resolve([{"chunk_id": "chunk-1"}], [result])

    assert resolution.citations == []
    assert resolution.failure_reason == "unresolved_citation"


def test_citation_resolver_rejects_unknown_source_type_value():
    results = [_result("chunk-1", 1, 0.9, source_type="trusted")]

    resolution = CitationResolver.resolve([{"chunk_id": "chunk-1"}], results)

    assert resolution.citations == []
    assert resolution.failure_reason == "unresolved_citation"


def test_citation_resolver_rejects_non_string_chunk_id():
    results = [_result("chunk-1", 1, 0.9)]

    resolution = CitationResolver.resolve([{"chunk_id": 7}], results)

    assert resolution.citations == []
    assert resolution.failure_reason == "unresolved_citation"


def test_citation_resolver_empty_llm_citations_returns_empty_set_without_failure():
    """Empty input is not a resolution failure — the caller (QueryUseCase)
    still refuses, but the distinction keeps the diagnostic reason honest."""
    results = [_result("chunk-1", 1, 0.9)]

    resolution = CitationResolver.resolve([], results)

    assert resolution.citations == []
    assert resolution.failure_reason is None


def _grounding_result(chunk_id: str, chunk_text: str, semantic_rank: int = 1) -> RetrievalResult:
    return _result(chunk_id, semantic_rank, 0.57, chunk_text=chunk_text)


_CHUNK_BODY = (
    "The quality control unit shall have the responsibility for approving or rejecting all "
    "components, drug product containers, closures, in-process materials, packaging material, "
    "labeling, and drug products."
)


def test_refusal_policy_binary_matches_legacy_is_confident():
    below = [_result("c", 1, 0.4)]
    at = [_result("c", 1, 0.5999)]
    assert RefusalPolicy(0.5999).classify_score(0.4) == "hard_refuse"
    assert RefusalPolicy(0.5999).classify(below) == "hard_refuse"
    assert RefusalPolicy(0.5999).classify(at) == "confident"
    assert RefusalPolicy(0.5999).classify_score(None) == "hard_refuse"


@pytest.mark.parametrize(
    "score, expected",
    [
        (None, "hard_refuse"),
        (0.4999, "hard_refuse"),
        (0.5499, "hard_refuse"),
        (0.5500, "grounded_review"),
        (0.5642, "grounded_review"),
        (0.5998, "grounded_review"),
        (0.5999, "confident"),
        (0.7, "confident"),
    ],
)
def test_refusal_policy_grounded_review_bands(score, expected):
    policy = RefusalPolicy(0.5999, mode="grounded_review", review_floor=0.5500)
    assert policy.classify_score(score) == expected


def test_refusal_policy_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        RefusalPolicy(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        RefusalPolicy(0.6, review_floor=float("inf"))


def test_refusal_policy_grounded_review_requires_floor_below_threshold():
    with pytest.raises(ValueError, match="strictly below"):
        RefusalPolicy(0.55, mode="grounded_review", review_floor=0.55)
    with pytest.raises(ValueError, match="strictly below"):
        RefusalPolicy(0.55, mode="grounded_review", review_floor=0.60)


def test_refusal_policy_hard_refuse_reason_depends_on_mode():
    assert RefusalPolicy(0.5999).hard_refuse_reason() == "below_binary_threshold"
    assert (
        RefusalPolicy(0.5999, mode="grounded_review").hard_refuse_reason() == "below_review_floor"
    )


def test_normalize_evidence_text_collapses_whitespace_only():
    assert normalize_evidence_text("a  b\n\tc ") == "a b c"


def test_grounded_evidence_accepts_exact_quote_with_collapsed_whitespace():
    results = [_grounding_result("c1", _CHUNK_BODY)]
    quote = "quality control unit shall\n  have the responsibility for approving or rejecting all components"
    out = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "c1", "supporting_quote": quote}], results
    )
    assert out.failure_reason is None
    assert [c.chunk_id for c in out.citations] == ["c1"]


def test_grounded_evidence_empty_is_missing_evidence():
    out = GroundedEvidenceResolver.resolve([], [_grounding_result("c1", _CHUNK_BODY)])
    assert out.failure_reason == "missing_evidence"
    assert out.citations == []


def test_grounded_evidence_shape_errors():
    results = [_grounding_result("c1", _CHUNK_BODY)]
    assert GroundedEvidenceResolver.resolve(["not-a-dict"], results).failure_reason == "invalid_evidence_shape"
    assert (
        GroundedEvidenceResolver.resolve([{"chunk_id": "c1"}], results).failure_reason
        == "invalid_evidence_shape"
    )
    assert (
        GroundedEvidenceResolver.resolve(
            [{"chunk_id": 5, "supporting_quote": "x" * 50}], results
        ).failure_reason
        == "invalid_evidence_shape"
    )


def test_grounded_evidence_chunk_not_retrieved():
    results = [_grounding_result("c1", _CHUNK_BODY)]
    out = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "other", "supporting_quote": _CHUNK_BODY[:60]}], results
    )
    assert out.failure_reason == "chunk_not_retrieved"


def test_grounded_evidence_quote_length_bounds():
    results = [_grounding_result("c1", _CHUNK_BODY + " " + "z" * 700)]
    short = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "c1", "supporting_quote": "too short"}], results
    )
    assert short.failure_reason == "quote_too_short"
    long_quote = "z" * 601
    results_long = [_grounding_result("c1", "z" * 800)]
    out_long = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "c1", "supporting_quote": long_quote}], results_long
    )
    assert out_long.failure_reason == "quote_too_long"


def test_grounded_evidence_quote_not_a_substring():
    results = [_grounding_result("c1", _CHUNK_BODY)]
    out = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "c1", "supporting_quote": "the QC unit paraphrased differently for at least forty chars"}],
        results,
    )
    assert out.failure_reason == "quote_not_found"


def test_grounded_evidence_fail_closed_on_any_bad_item():
    results = [_grounding_result("c1", _CHUNK_BODY)]
    good = {"chunk_id": "c1", "supporting_quote": _CHUNK_BODY[:80]}
    bad = {"chunk_id": "c1", "supporting_quote": "nope " * 20}
    out = GroundedEvidenceResolver.resolve([good, bad], results)
    assert out.failure_reason == "quote_not_found"
    assert out.citations == []


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
