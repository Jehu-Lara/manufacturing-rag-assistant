from __future__ import annotations

import json

from api.messages import REFUSAL_MESSAGE

VALID_LANGUAGES = ("en", "es")

# chunk_id-only: full citation fields (document_title, section_heading,
# revision, document_id) are resolved server-side from real retrieved-chunk
# metadata elsewhere (api/generation.py), never trusted from LLM output —
# this prevents the model from hallucinating citation metadata.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
        "refused": {"type": "boolean"},
    },
    "required": ["answer", "citations", "refused"],
    "additionalProperties": False,
}


def build_system_prompt(language: str) -> str:
    """Design choice (a): the instructional rules are always written in
    English regardless of `language` — only rule 5's directive word
    (English/Spanish) varies. Rationale: these are instructions to the model,
    not user-facing content, so there's no requirement they be understood in
    Spanish to produce a Spanish answer; keeping one English template avoids
    maintaining and keeping in sync a second, fully-translated prompt whose
    rules could drift from the English version over time."""
    if language not in VALID_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}. Must be one of {VALID_LANGUAGES}.")

    refusal_message = REFUSAL_MESSAGE[language]
    respond_in = "English" if language == "en" else "Spanish"

    return (
        "You are a manufacturing knowledge assistant. Follow these rules exactly:\n"
        "1. Answer ONLY using the retrieved context chunks provided in the user message. "
        "Do not use any outside knowledge.\n"
        "2. If the retrieved chunks do not contain enough information to answer confidently, "
        'set "refused" to true, leave "citations" empty, and set "answer" to exactly this string '
        f'(do not translate or alter it): "{refusal_message}"\n'
        "3. Never state anything that is not directly supported by a retrieved chunk.\n"
        '4. For every claim in your answer, cite the chunk_id(s) of every retrieved chunk that '
        'supports it, in the "citations" array.\n'
        f'5. Respond in {respond_in}: write the "answer" field in {respond_in}. This only affects '
        'the language of the "answer" field — any chunk_id you cite still refers to the chunk '
        "exactly as given; the underlying document and section text is always in English, "
        "regardless of the answer's language.\n"
        "6. Output ONLY valid JSON matching this JSON Schema, with no other text before or after "
        f"it:\n{json.dumps(JSON_SCHEMA)}\n"
    )


def build_user_prompt(question: str, retrieved_chunks: list) -> str:
    lines = [f"Question: {question}", "", "Retrieved context chunks:"]
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.metadata
        lines.append(
            f"{index}. chunk_id: {chunk.chunk_id}\n"
            f"   document_title: {metadata['document_title']}\n"
            f"   section_heading: {metadata['section_heading']}\n"
            f"   revision: {metadata['revision']}\n"
            f"   text: {metadata['chunk_text']}"
        )
    return "\n".join(lines)
