from __future__ import annotations

import uuid

import streamlit as st

from src.web import client
from src.web.i18n import EXAMPLE_ANSWERABLE_QUESTION, EXAMPLE_UNANSWERABLE_QUESTION, UI_LABELS
from src.web.render import render_result


def _submit(question: str, lang: str) -> None:
    st.session_state["last_response"] = None
    st.session_state["last_error"] = None
    try:
        response = client.query(question, lang, session_id=st.session_state.get("client_session_id"))
        response.raise_for_status()
        st.session_state["last_response"] = response.json()
    except Exception as exc:
        st.session_state["last_error"] = str(exc)


def _fill_and_submit_example(question_text: str) -> None:
    st.session_state["query_text"] = question_text
    _submit(question_text, st.session_state["language"])


def main() -> None:
    st.set_page_config(page_title="Manufacturing Knowledge Assistant", page_icon="🏭")

    # One opaque id per browser session, minted here and never persisted. It
    # exists only so the API can rate-limit per visitor: every question reaches
    # FastAPI from this same Streamlit process over loopback, so without it all
    # visitors would share one bucket. Deliberately not tied to the language
    # toggle or the query text, so switching language cannot mint a new bucket.
    if "client_session_id" not in st.session_state:
        st.session_state["client_session_id"] = str(uuid.uuid4())
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
    st.warning(labels["privacy_warning"])

    is_ready = client.ready()

    question = st.text_input(
        labels["question_input_label"],
        placeholder=labels["question_input_placeholder"],
        key="query_text",
    )
    ask_clicked = st.button(labels["ask_button"], disabled=not is_ready)
    if not is_ready:
        st.caption(labels["not_ready_label"])

    st.subheader(labels["try_example_heading"])
    col1, col2 = st.columns(2)
    col1.button(
        labels["try_example_answerable_button"],
        on_click=_fill_and_submit_example,
        args=(EXAMPLE_ANSWERABLE_QUESTION,),
        disabled=not is_ready,
    )
    col2.button(
        labels["try_example_unanswerable_button"],
        on_click=_fill_and_submit_example,
        args=(EXAMPLE_UNANSWERABLE_QUESTION,),
        disabled=not is_ready,
    )

    if ask_clicked and question:
        _submit(question, lang)

    render_result(labels, lang, st.session_state["last_response"], st.session_state["last_error"])


if __name__ == "__main__":
    main()
