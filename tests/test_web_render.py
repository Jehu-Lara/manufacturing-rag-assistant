from __future__ import annotations

import pytest

from src.web.i18n import UI_LABELS
from src.web.render import _gate_caption, classify_response_state, format_citations


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


def test_gate_caption_binary_hides_review_floor_shows_band():
    labels = UI_LABELS["en"]
    caption = _gate_caption(
        labels,
        {"confidence": 0.91, "threshold": 0.5999, "review_floor": None, "gate_band": "confident"},
    )
    assert "0.5999" in caption
    assert labels["review_floor_label"] not in caption
    assert "confident" in caption


def test_gate_caption_grounded_shows_both_limits_and_band():
    labels = UI_LABELS["en"]
    caption = _gate_caption(
        labels,
        {
            "confidence": 0.57,
            "threshold": 0.5999,
            "review_floor": 0.55,
            "gate_band": "grounded_review",
        },
    )
    assert labels["review_floor_label"] in caption
    assert "grounded_review" in caption
