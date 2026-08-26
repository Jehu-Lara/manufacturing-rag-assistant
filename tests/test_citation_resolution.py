from __future__ import annotations

import logging

from api.generation import _resolve_citations
from api.schemas import Citation
from retrieval.hybrid import RetrievalResult


def _result(chunk_id: str, **metadata_overrides: str) -> RetrievalResult:
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
        fused_score=0.5,
        semantic_rank=1,
        semantic_score=0.9,
        bm25_rank=1,
        bm25_score=1.0,
        metadata=metadata,
    )


def test_all_cited_chunk_ids_found_builds_correct_citations():
    results = [_result("chunk-1"), _result("chunk-2")]
    llm_citations = [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}]

    resolved = _resolve_citations(llm_citations, results)

    assert resolved == [
        Citation(
            document_id="doc-chunk-1",
            document_title="Title for chunk-1",
            section_heading="Section for chunk-1",
            revision="Rev A",
            chunk_id="chunk-1",
        ),
        Citation(
            document_id="doc-chunk-2",
            document_title="Title for chunk-2",
            section_heading="Section for chunk-2",
            revision="Rev A",
            chunk_id="chunk-2",
        ),
    ]


def test_cited_chunk_id_not_in_results_is_dropped_and_logs_warning(caplog):
    results = [_result("chunk-1")]
    llm_citations = [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-unknown"}]

    with caplog.at_level(logging.WARNING, logger="api.generation"):
        resolved = _resolve_citations(llm_citations, results)

    assert len(resolved) == 1
    assert resolved[0].chunk_id == "chunk-1"
    assert any(
        record.__dict__.get("event") == "citation_not_in_retrieved_set"
        and record.__dict__.get("chunk_id") == "chunk-unknown"
        for record in caplog.records
    )


def test_empty_llm_citations_returns_empty_list_no_error():
    results = [_result("chunk-1")]

    resolved = _resolve_citations([], results)

    assert resolved == []
