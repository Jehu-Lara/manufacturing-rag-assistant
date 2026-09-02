from __future__ import annotations

# Duplicated from api.messages.UI_LABELS, not imported (Resolved Decision #6):
# src/web/ must have zero imports of src.domain/features/adapters, and
# api.messages lives alongside those in spirit even though physically it's
# still a flat top-level module — this keeps web/ genuinely HTTP-only and
# import-isolated from the backend, matching the same isolation the
# import-invariant test enforces for src/domain. One small, deliberate,
# documented data duplication (a string catalog, not logic).
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
        "review_floor_label": "Review floor",
        "gate_band_label": "Gate band",
        "refused_badge": "Not answered",
        "refused_heading": "Insufficient information",
        "error_badge": "Error",
        "error_heading": "Something went wrong",
        "try_example_heading": "Try an example",
        "try_example_answerable_button": "Try an answerable question",
        "try_example_unanswerable_button": "Try an unanswerable question",
        "backend_unreachable_label": "Could not reach the backend",
        "request_id_label": "Request ID",
        "not_ready_label": "Backend index is still loading — try again shortly.",
        "privacy_warning": (
            "Do not enter confidential, personal, regulated, or proprietary information. "
            "Your question and retrieved context are sent to the configured external LLM provider."
        ),
        # Shown on every answered response: the corpus mixes real regulatory
        # text with clearly-labeled synthetic examples, and an answer here is
        # a reading aid, never the controlling document on a plant floor.
        "safety_notice": (
            "AI-generated reference only. Verify against the controlling SOP and your "
            "facility's lockout/tagout (LOTO) or energy-control procedure before acting."
        ),
        "synthetic_source_badge": "**[SYNTHETIC / EXAMPLE]**",
    },
    "es": {
        "title": "Asistente de Conocimiento de Manufactura",
        "subtitle": (
            "Haga preguntas sobre regulaciones de manufactura, "
            "procedimientos operativos y normas de seguridad."
        ),
        "language_toggle_label": "Idioma",
        "question_input_label": "Su pregunta",
        "question_input_placeholder": "p. ej. ¿Cuáles son las responsabilidades de la unidad de control de calidad?",
        "ask_button": "Preguntar",
        "answer_heading": "Respuesta",
        "citations_heading": "Fuentes",
        "confidence_label": "Puntaje de confianza",
        "threshold_label": "Umbral de negativa",
        "review_floor_label": "Piso de revisión",
        "gate_band_label": "Banda de decisión",
        "refused_badge": "No respondida",
        "refused_heading": "Información insuficiente",
        "error_badge": "Error",
        "error_heading": "Ocurrió un problema",
        "try_example_heading": "Probar un ejemplo",
        "try_example_answerable_button": "Probar una pregunta respondible",
        "try_example_unanswerable_button": "Probar una pregunta no respondible",
        "backend_unreachable_label": "No se pudo contactar al servidor",
        "request_id_label": "ID de solicitud",
        "not_ready_label": "El índice del servidor aún se está cargando — intente de nuevo en un momento.",
        "privacy_warning": (
            "No introduzca información confidencial, personal, regulada ni propietaria. "
            "Su pregunta y el contexto recuperado se envían al proveedor LLM externo configurado."
        ),
        "safety_notice": (
            "Referencia generada por IA únicamente. Verifique la respuesta contra el SOP "
            "vigente y el procedimiento de bloqueo/etiquetado (LOTO) o control de energía "
            "de su planta antes de actuar."
        ),
        "synthetic_source_badge": "**[SINTÉTICO / EJEMPLO]**",
    },
}

# Hardcoded (Resolved Decision #7): src/web/ must not read eval_set.json
# directly. These are real questions drawn from the eval set at the time
# this was written (one answerable, one unanswerable, English), not
# regenerated from the live eval set — purely for UX demo buttons, not an
# eval-integrity concern (that guarantee lives entirely in
# src/features/evaluation).
EXAMPLE_ANSWERABLE_QUESTION = (
    "What must an energy-control procedure include according to OSHA's lockout/tagout requirements?"
)
EXAMPLE_UNANSWERABLE_QUESTION = (
    "What are the qualitative and quantitative respirator fit-testing protocols required "
    "before an employee is assigned a respirator?"
)
