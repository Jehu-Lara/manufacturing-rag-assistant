from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import httpx
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.messages import UI_LABELS  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
EVAL_SET_PATH = _REPO_ROOT / "eval" / "eval_set.json"
REQUEST_TIMEOUT_SECONDS = 60.0


def classify_response_state(response_json: dict) -> Literal["answer", "refused", "error"]:
    """status=="error" wins regardless of refused, since a technical failure
    can leave refused in any state — check it first."""
    if response_json["status"] == "error":
        return "error"
    if response_json["refused"]:
        return "refused"
    return "answer"


def format_citations(citations: list[dict], lang: str) -> str:
    if not citations:
        return ""
    lines = [
        f"- **{citation['document_title']}** — {citation['section_heading']} (rev. {citation['revision']})"
        for citation in citations
    ]
    return "\n".join(lines)


def pick_example_questions(eval_set_path: Path) -> tuple[dict, dict]:
    with eval_set_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    en_questions = [q for q in data["questions"] if q["language"] == "en"]
    answerable = next((q for q in en_questions if q["answerable"] is True), None)
    unanswerable = next((q for q in en_questions if q["answerable"] is False), None)

    if answerable is None or unanswerable is None:
        raise ValueError(
            f"{eval_set_path} must contain at least one EN answerable and one "
            "EN unanswerable question for the UI example buttons"
        )

    return answerable, unanswerable


def _submit(question: str, lang: str) -> None:
    st.session_state["last_response"] = None
    st.session_state["last_error"] = None
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "language": lang},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        st.session_state["last_response"] = response.json()
    except httpx.HTTPError as exc:
        st.session_state["last_error"] = str(exc)


def _fill_and_submit_example(question_text: str) -> None:
    st.session_state["query_text"] = question_text
    _submit(question_text, st.session_state["language"])


def _render_result(labels: dict[str, str], lang: str) -> None:
    if st.session_state["last_error"] is not None:
        st.error(
            f"{'Could not reach the backend' if lang == 'en' else 'No se pudo contactar al servidor'}: "
            f"{st.session_state['last_error']}"
        )
        return

    response_json = st.session_state["last_response"]
    if response_json is None:
        return

    state = classify_response_state(response_json)

    if state == "answer":
        st.subheader(labels["answer_heading"])
        st.write(response_json["answer"])
        st.subheader(labels["citations_heading"])
        st.markdown(format_citations(response_json["citations"], lang))
        confidence = response_json.get("confidence")
        threshold = response_json.get("threshold")
        st.caption(f"{labels['confidence_label']}: {confidence} | {labels['threshold_label']}: {threshold}")
    elif state == "refused":
        st.warning(f"**{labels['refused_badge']}** — {labels['refused_heading']}")
        st.write(response_json["answer"])
    else:
        st.error(f"**{labels['error_badge']}** — {labels['error_heading']}")
        st.write(response_json["answer"])


def main() -> None:
    st.set_page_config(page_title="Manufacturing Knowledge Assistant", page_icon="🏭")

    if "language" not in st.session_state:
        st.session_state["language"] = "en"
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None

    lang_options = ["en", "es"]
    st.radio(
        UI_LABELS[st.session_state["language"]]["language_toggle_label"],
        options=lang_options,
        format_func=lambda code: "English" if code == "en" else "Español",
        horizontal=True,
        key="language",
    )
    lang = st.session_state["language"]
    labels = UI_LABELS[lang]

    st.title(labels["title"])
    st.caption(labels["subtitle"])

    question = st.text_input(
        labels["question_input_label"],
        placeholder=labels["question_input_placeholder"],
        key="query_text",
    )
    ask_clicked = st.button(labels["ask_button"])

    st.subheader(labels["try_example_heading"])
    answerable_q, unanswerable_q = pick_example_questions(EVAL_SET_PATH)
    col1, col2 = st.columns(2)
    col1.button(
        labels["try_example_answerable_button"],
        on_click=_fill_and_submit_example,
        args=(answerable_q["question"],),
    )
    col2.button(
        labels["try_example_unanswerable_button"],
        on_click=_fill_and_submit_example,
        args=(unanswerable_q["question"],),
    )

    if ask_clicked and question:
        _submit(question, lang)

    _render_result(labels, lang)


if __name__ == "__main__":
    main()
