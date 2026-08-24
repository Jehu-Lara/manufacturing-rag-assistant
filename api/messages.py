from __future__ import annotations

REFUSAL_MESSAGE: dict[str, str] = {
    "en": (
        "I don't have enough information in the available documents to answer "
        "this confidently. Try rephrasing your question with more specific "
        "terms, or consult a subject-matter expert for a definitive answer."
    ),
    "es": (
        "No cuento con suficiente información en los documentos disponibles "
        "para responder esto con confianza. Intente reformular su pregunta con "
        "términos más específicos, o consulte a un experto en la materia para "
        "obtener una respuesta definitiva."
    ),
}

GENERATION_ERROR_MESSAGE: dict[str, str] = {
    "en": (
        "A technical error occurred while generating this answer. This is not "
        "a refusal due to insufficient information — please try again in a "
        "moment."
    ),
    "es": (
        "Ocurrió un error técnico al generar esta respuesta. Esto no es una "
        "negativa por falta de información — por favor, inténtelo de nuevo en "
        "un momento."
    ),
}

UI_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "title": "Manufacturing Knowledge Assistant",
        "subtitle": "Ask questions about manufacturing regulations, SOPs, and safety procedures.",
        "language_toggle_label": "Language",
        "question_input_label": "Your question",
        "question_input_placeholder": "e.g. What are the responsibilities of the quality control unit?",
        "ask_button": "Ask",
        "answer_heading": "Answer",
        "citations_heading": "Sources",
        "confidence_label": "Confidence score",
        "threshold_label": "Refusal threshold",
        "refused_badge": "Not answered",
        "refused_heading": "Insufficient information",
        "error_badge": "Error",
        "error_heading": "Something went wrong",
        "try_example_heading": "Try an example",
        "try_example_answerable_button": "Try an answerable question",
        "try_example_unanswerable_button": "Try an unanswerable question",
        "backend_unreachable_label": "Could not reach the backend",
    },
    "es": {
        "title": "Asistente de Conocimiento de Manufactura",
        "subtitle": "Haga preguntas sobre regulaciones de manufactura, procedimientos operativos y normas de seguridad.",
        "language_toggle_label": "Idioma",
        "question_input_label": "Su pregunta",
        "question_input_placeholder": "p. ej. ¿Cuáles son las responsabilidades de la unidad de control de calidad?",
        "ask_button": "Preguntar",
        "answer_heading": "Respuesta",
        "citations_heading": "Fuentes",
        "confidence_label": "Puntaje de confianza",
        "threshold_label": "Umbral de negativa",
        "refused_badge": "No respondida",
        "refused_heading": "Información insuficiente",
        "error_badge": "Error",
        "error_heading": "Ocurrió un problema",
        "try_example_heading": "Probar un ejemplo",
        "try_example_answerable_button": "Probar una pregunta respondible",
        "try_example_unanswerable_button": "Probar una pregunta no respondible",
        "backend_unreachable_label": "No se pudo contactar al servidor",
    },
}
