from __future__ import annotations

from src.features.query.prompts import GENERATION_ERROR_MESSAGE, REFUSAL_MESSAGE
from src.web.i18n import EXAMPLE_ANSWERABLE_QUESTION, EXAMPLE_UNANSWERABLE_QUESTION, UI_LABELS

_REQUIRED_UI_LABEL_KEYS = (
    "title",
    "subtitle",
    "language_toggle_label",
    "question_input_label",
    "question_input_placeholder",
    "ask_button",
    "answer_heading",
    "citations_heading",
    "confidence_label",
    "threshold_label",
    "review_floor_label",
    "gate_band_label",
    "refused_badge",
    "refused_heading",
    "error_badge",
    "error_heading",
    "try_example_heading",
    "try_example_answerable_button",
    "try_example_unanswerable_button",
    "backend_unreachable_label",
    "request_id_label",
    "not_ready_label",
    "privacy_warning",
)


def test_ui_labels_present_in_both_languages() -> None:
    for language in ("en", "es"):
        assert language in UI_LABELS
        for key in _REQUIRED_UI_LABEL_KEYS:
            assert key in UI_LABELS[language], f"missing '{key}' in UI_LABELS[{language!r}]"
            assert UI_LABELS[language][key].strip()


def test_refusal_and_error_messages_are_non_empty_and_distinct() -> None:
    for language in ("en", "es"):
        assert REFUSAL_MESSAGE[language].strip()
        assert GENERATION_ERROR_MESSAGE[language].strip()
        assert REFUSAL_MESSAGE[language] != GENERATION_ERROR_MESSAGE[language]


def test_example_questions_are_non_empty_and_distinct() -> None:
    assert EXAMPLE_ANSWERABLE_QUESTION.strip()
    assert EXAMPLE_UNANSWERABLE_QUESTION.strip()
    assert EXAMPLE_ANSWERABLE_QUESTION != EXAMPLE_UNANSWERABLE_QUESTION
