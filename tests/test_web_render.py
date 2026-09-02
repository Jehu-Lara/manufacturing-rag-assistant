from __future__ import annotations

import pytest

from src.web import render as render_module
from src.web.i18n import UI_LABELS
from src.web.render import (
    _gate_caption,
    classify_response_state,
    format_citations,
    render_result,
)


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


def _citation(source_type: str, title: str = "Doc A") -> dict:
    return {
        "document_id": "doc-a",
        "document_title": title,
        "section_heading": "Section A",
        "revision": "1",
        "chunk_id": "doc-a::chunk-0001",
        "source_type": source_type,
    }


def test_format_citations_marks_synthetic_source_only():
    result = format_citations([_citation("public", "Public Doc"), _citation("synthetic", "Synthetic Doc")], "en")
    public_line, synthetic_line = result.splitlines()

    assert UI_LABELS["en"]["synthetic_source_badge"] not in public_line
    assert UI_LABELS["en"]["synthetic_source_badge"] in synthetic_line


def test_format_citations_synthetic_badge_follows_language():
    result = format_citations([_citation("synthetic")], "es")
    assert UI_LABELS["es"]["synthetic_source_badge"] in result
    assert UI_LABELS["en"]["synthetic_source_badge"] not in result


class _StCapture:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.writes: list[str] = []

    def warning(self, text):
        self.warnings.append(text)

    def error(self, text):
        pass

    def write(self, text):
        self.writes.append(text)

    def subheader(self, text):
        pass

    def markdown(self, text):
        pass

    def caption(self, text):
        pass


def _capture_render(monkeypatch, response: dict) -> _StCapture:
    capture = _StCapture()
    monkeypatch.setattr(render_module, "st", capture)
    render_result(UI_LABELS["en"], "en", response, None)
    return capture


def test_render_result_shows_safety_notice_on_an_answered_response(monkeypatch):
    capture = _capture_render(
        monkeypatch,
        {
            "status": "ok",
            "refused": False,
            "answer": "Lock the energy source.",
            "citations": [_citation("public")],
            "confidence": 0.9,
            "threshold": 0.5999,
            "review_floor": None,
            "gate_band": "confident",
            "request_id": "req-1",
        },
    )

    assert UI_LABELS["en"]["safety_notice"] in capture.warnings


def test_render_result_omits_safety_notice_on_refusal(monkeypatch):
    """A refusal carries no generated content to verify — the notice would only
    dilute the one that matters on real answers."""
    capture = _capture_render(
        monkeypatch,
        {
            "status": "ok",
            "refused": True,
            "answer": "I don't have enough information...",
            "citations": [],
            "request_id": "req-2",
        },
    )

    assert UI_LABELS["en"]["safety_notice"] not in capture.warnings
