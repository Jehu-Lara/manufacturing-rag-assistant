from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.config import Settings
from api.main import app
from api.schemas import QueryResponse
from retrieval.hybrid import RetrievalResult


def _client() -> TestClient:
    return TestClient(app)


def _answerable_response() -> QueryResponse:
    return QueryResponse(
        answer="The QC unit is responsible for approving or rejecting components.",
        citations=[
            {
                "document_id": "doc-1",
                "document_title": "21 CFR Part 211",
                "section_heading": "§ 211.22 Responsibilities of quality control unit",
                "revision": "Rev A",
                "chunk_id": "chunk-1",
            }
        ],
        refused=False,
        status="ok",
        confidence=0.81,
        threshold=0.5599,
        language="en",
        request_id="11111111-1111-1111-1111-111111111111",
    )


def _refused_response() -> QueryResponse:
    return QueryResponse(
        answer="I don't have enough information in the corpus to answer that confidently.",
        citations=[],
        refused=True,
        status="ok",
        confidence=0.12,
        threshold=0.5599,
        language="en",
        request_id="22222222-2222-2222-2222-222222222222",
    )


def test_health_returns_200_with_index_loaded_true_when_collection_reachable():
    with patch("retrieval.vector_store.get_collection", return_value=MagicMock()):
        with _client() as client:
            response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["embedding_model"], str) and body["embedding_model"]
    assert body["llm_provider_primary"] in ("groq", "openai")
    assert body["index_loaded"] is True


def test_health_returns_200_with_index_loaded_false_when_collection_unreachable():
    with patch("retrieval.vector_store.get_collection", side_effect=RuntimeError("no such collection")):
        with _client() as client:
            response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is False


@patch("api.generation.answer_question")
def test_query_answerable_returns_200_and_matches_mock(mock_answer_question):
    mock_answer_question.return_value = _answerable_response()

    with _client() as client:
        response = client.post("/query", json={"question": "What is the QC unit responsible for?", "language": "en"})

    assert response.status_code == 200
    assert response.json() == _answerable_response().model_dump()


@patch("api.generation.answer_question")
def test_query_refused_returns_200_and_matches_mock(mock_answer_question):
    mock_answer_question.return_value = _refused_response()

    with _client() as client:
        response = client.post("/query", json={"question": "What's the weather today?", "language": "en"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body == _refused_response().model_dump()


@patch("api.generation.answer_question")
def test_query_missing_question_returns_422(mock_answer_question):
    with _client() as client:
        response = client.post("/query", json={"language": "en"})

    assert response.status_code == 422
    mock_answer_question.assert_not_called()


@patch("api.generation.answer_question")
def test_query_invalid_language_returns_422(mock_answer_question):
    with _client() as client:
        response = client.post("/query", json={"question": "What is the SOP?", "language": "fr"})

    assert response.status_code == 422
    mock_answer_question.assert_not_called()


@patch("api.generation.answer_question")
def test_query_calls_answer_question_with_request_body_values(mock_answer_question):
    mock_answer_question.return_value = _answerable_response()

    with _client() as client:
        client.post("/query", json={"question": "What is the QC unit responsible for?", "language": "es"})

    args, kwargs = mock_answer_question.call_args
    assert args[0] == "What is the QC unit responsible for?"
    assert args[1] == "es"


@patch("api.main.load_settings")
@patch("api.llm_client.groq.Groq")
@patch("retrieval.hybrid.retrieve")
def test_query_end_to_end_resolves_citations_from_retrieved_metadata_not_llm_payload(
    mock_retrieve, mock_groq_cls, mock_load_settings
):
    """Composition test: HTTP -> refusal gate -> prompt build -> LLM client
    parse/validate -> citation resolution -> response serialization, mocking
    only the true external boundaries (retrieval, the provider SDK client) —
    not api.generation.answer_question or api.llm_client.generate_structured.
    The LLM's JSON payload can only ever carry a chunk_id (JSON_SCHEMA forbids
    any other citation property), so a response citation exposing
    document_title/section_heading/revision/document_id proves those came
    from the retrieved chunk's real metadata, not from the LLM output."""
    mock_load_settings.return_value = Settings(
        groq_api_key="fake-key",
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=0.3,
        log_level="INFO",
    )

    retrieved = RetrievalResult(
        chunk_id="chunk-abc",
        fused_score=1.0,
        semantic_rank=1,
        semantic_score=0.9,
        bm25_rank=1,
        bm25_score=1.0,
        metadata={
            "document_id": "doc-real",
            "document_title": "Real Retrieved Title",
            "section_heading": "Real Section",
            "revision": "Rev Z",
            "chunk_id": "chunk-abc",
            "chunk_text": "some real chunk text",
        },
    )
    mock_retrieve.return_value = [retrieved]

    llm_payload = {
        "answer": "The QC unit is responsible for X.",
        "citations": [{"chunk_id": "chunk-abc"}],
        "refused": False,
    }
    message = MagicMock()
    message.content = json.dumps(llm_payload)
    choice = MagicMock()
    choice.message = message
    fake_response = MagicMock()
    fake_response.choices = [choice]

    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.return_value = fake_response
    mock_groq_cls.return_value = mock_groq_client

    with _client() as client:
        response = client.post(
            "/query", json={"question": "What is the QC unit responsible for?", "language": "en"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["status"] == "ok"
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["chunk_id"] == "chunk-abc"
    assert citation["document_id"] == "doc-real"
    assert citation["document_title"] == "Real Retrieved Title"
    assert citation["section_heading"] == "Real Section"
    assert citation["revision"] == "Rev Z"
