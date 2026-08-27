from __future__ import annotations

import pytest

from src.web.render import classify_response_state, format_citations


def test_classify_response_state_normal_answer():
    response = {"status": "ok", "refused": False, "answer": "The QC unit is responsible for..."}
    assert classify_response_state(response) == "answer"


def test_classify_response_state_refused():
    response = {"status": "ok", "refused": True, "answer": "I don't have enough information..."}
    assert classify_response_state(response) == "refused"


def test_classify_response_state_error():
    response = {"status": "error", "refused": False, "answer": "A technical error occurred..."}
    assert classify_response_state(response) == "error"


def test_classify_response_state_error_wins_even_if_refused_true():
    response = {"status": "error", "refused": True, "answer": "..."}
    assert classify_response_state(response) == "error"


def test_classify_response_state_missing_keys_raises():
    with pytest.raises(KeyError):
        classify_response_state({"refused": False})


def test_format_citations_uses_human_readable_fields_not_ids():
    citations = [
        {
            "document_id": "osha-3120-lockout-tagout",
            "document_title": "OSHA 3120 — Control of Hazardous Energy",
            "section_heading": "Commonly Used Terms",
            "revision": "2018",
            "chunk_id": "osha-3120-lockout-tagout::chunk-0018",
        }
    ]
    result = format_citations(citations, "en")
    assert "OSHA 3120 — Control of Hazardous Energy" in result
    assert "Commonly Used Terms" in result
    assert "2018" in result
    assert "osha-3120-lockout-tagout::chunk-0018" not in result
    assert "osha-3120-lockout-tagout" not in result.replace("Control of Hazardous Energy", "")


def test_format_citations_multiple_entries_one_line_each():
    citations = [
        {
            "document_id": "doc-a",
            "document_title": "Doc A",
            "section_heading": "Section A",
            "revision": "1",
            "chunk_id": "doc-a::chunk-0001",
        },
        {
            "document_id": "doc-b",
            "document_title": "Doc B",
            "section_heading": "Section B",
            "revision": "2",
            "chunk_id": "doc-b::chunk-0002",
        },
    ]
    result = format_citations(citations, "en")
    assert len(result.splitlines()) == 2


def test_format_citations_empty_list_returns_empty_string():
    assert format_citations([], "en") == ""
