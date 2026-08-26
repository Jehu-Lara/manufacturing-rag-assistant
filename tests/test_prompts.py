from __future__ import annotations

import json

import pytest

from api.messages import REFUSAL_MESSAGE
from api.prompts import JSON_SCHEMA, build_system_prompt, build_user_prompt
from retrieval.hybrid import RetrievalResult


def test_json_schema_top_level_shape():
    assert JSON_SCHEMA["type"] == "object"
    assert set(JSON_SCHEMA["required"]) == {"answer", "citations", "refused"}
    assert JSON_SCHEMA["additionalProperties"] is False


def test_json_schema_properties():
    properties = JSON_SCHEMA["properties"]
    assert properties["answer"] == {"type": "string"}
    assert properties["refused"] == {"type": "boolean"}
    assert properties["citations"]["type"] == "array"


def test_json_schema_citations_items_shape():
    items = JSON_SCHEMA["properties"]["citations"]["items"]
    assert items["type"] == "object"
    assert items["properties"] == {"chunk_id": {"type": "string"}}
    assert items["required"] == ["chunk_id"]
    assert items["additionalProperties"] is False


def test_build_system_prompt_en_contains_english_refusal_not_spanish():
    prompt = build_system_prompt("en")
    assert REFUSAL_MESSAGE["en"] in prompt
    assert REFUSAL_MESSAGE["es"] not in prompt


def test_build_system_prompt_es_contains_spanish_refusal_not_english():
    prompt = build_system_prompt("es")
    assert REFUSAL_MESSAGE["es"] in prompt
    assert REFUSAL_MESSAGE["en"] not in prompt


def test_build_system_prompt_invalid_language_raises():
    with pytest.raises(ValueError):
        build_system_prompt("fr")


def test_build_system_prompt_embeds_json_schema():
    prompt = build_system_prompt("en")
    assert "additionalProperties" in prompt
    assert json.dumps(JSON_SCHEMA) in prompt


def test_build_system_prompt_es_also_embeds_json_schema():
    prompt = build_system_prompt("es")
    assert json.dumps(JSON_SCHEMA) in prompt


def _fake_result(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=0.5,
        semantic_rank=1,
        semantic_score=0.9,
        bm25_rank=1,
        bm25_score=1.2,
        metadata={
            "document_title": f"Document for {chunk_id}",
            "section_heading": f"Section for {chunk_id}",
            "revision": "2026-01-01",
            "chunk_text": text,
        },
    )


def test_build_user_prompt_includes_question_chunk_ids_and_text():
    chunks = [
        _fake_result("doc-a__0001", "Quality control unit responsibilities text."),
        _fake_result("doc-b__0002", "Personnel qualifications text."),
        _fake_result("doc-c__0003", "Equipment maintenance text."),
    ]
    prompt = build_user_prompt("What are QC unit responsibilities?", chunks)

    assert "What are QC unit responsibilities?" in prompt
    for chunk in chunks:
        assert chunk.chunk_id in prompt
        assert chunk.metadata["chunk_text"] in prompt
