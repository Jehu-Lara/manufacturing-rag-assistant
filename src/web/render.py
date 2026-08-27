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
        confidence = last_response.get("confidence")
        threshold = last_response.get("threshold")
        st.caption(f"{labels['confidence_label']}: {confidence} | {labels['threshold_label']}: {threshold}")
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
