from __future__ import annotations

from typing import Any, Literal, Optional

import streamlit as st


def classify_response_state(response_json: dict[str, Any]) -> Literal["answer", "refused", "error"]:
    """status=="error" wins regardless of refused, since a technical failure
    can leave refused in any state — check it first."""
    if response_json["status"] == "error":
        return "error"
    if response_json["refused"]:
        return "refused"
    return "answer"


def format_citations(citations: list[dict[str, Any]], lang: str) -> str:
    if not citations:
        return ""
    lines = [
        f"- **{citation['document_title']}** — {citation['section_heading']} (rev. {citation['revision']})"
        for citation in citations
    ]
    return "\n".join(lines)


def _gate_caption(labels: dict[str, str], response: dict[str, Any]) -> str:
    """Technical legend only. Shows both gate limits and the band the answer
    came from, so a grounded-review answer is never presented as high
    confidence."""
    parts = [
        f"{labels['confidence_label']}: {response.get('confidence')}",
        f"{labels['threshold_label']}: {response.get('threshold')}",
    ]
    review_floor = response.get("review_floor")
    if review_floor is not None:
        parts.append(f"{labels['review_floor_label']}: {review_floor}")
    gate_band = response.get("gate_band")
    if gate_band:
        parts.append(f"{labels['gate_band_label']}: {gate_band}")
    return " | ".join(parts)


def render_result(
    labels: dict[str, str], lang: str, last_response: Optional[dict[str, Any]], last_error: Optional[str]
) -> None:
    if last_error is not None:
        st.error(f"{labels['backend_unreachable_label']}: {last_error}")
        return

    if last_response is None:
        return

    state = classify_response_state(last_response)

    if state == "answer":
        st.subheader(labels["answer_heading"])
        st.write(last_response["answer"])
        st.subheader(labels["citations_heading"])
        st.markdown(format_citations(last_response["citations"], lang))
        st.caption(_gate_caption(labels, last_response))
    elif state == "refused":
        st.warning(f"**{labels['refused_badge']}** — {labels['refused_heading']}")
        st.write(last_response["answer"])
    else:
        st.error(f"**{labels['error_badge']}** — {labels['error_heading']}")
        st.write(last_response["answer"])

    # New, evidence-based addition: the API already returns request_id, the
    # UI just never displayed it — genuinely useful for support/debugging
    # correlation against backend logs (verified this was actually missing
    # before adding it).
    request_id = last_response.get("request_id")
    if request_id:
        st.caption(f"{labels['request_id_label']}: {request_id}")
