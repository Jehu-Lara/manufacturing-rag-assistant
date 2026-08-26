from __future__ import annotations

import uuid
from unittest.mock import patch

from api.config import Settings
from api.generation import answer_question
from api.llm_client import GenerationError
from api.messages import GENERATION_ERROR_MESSAGE, REFUSAL_MESSAGE
from api.schemas import Citation
from retrieval.hybrid import SEMANTIC_EXTRACTION_K, RetrievalResult

THRESHOLD = 0.5599


def _settings() -> Settings:
    return Settings(
        groq_api_key="groq-test-key",
        openai_api_key="openai-test-key",
        llm_provider="groq",
        refusal_cosine_threshold=THRESHOLD,
        log_level="INFO",
    )


def _result(chunk_id: str, semantic_rank: int, semantic_score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=1.0,
        semantic_rank=semantic_rank,
        semantic_score=semantic_score,
        bm25_rank=semantic_rank,
        bm25_score=1.0,
        metadata={
            "document_id": f"doc-{chunk_id}",
            "document_title": f"Title {chunk_id}",
            "section_heading": f"Section {chunk_id}",
            "revision": "Rev A",
            "chunk_id": chunk_id,
            "chunk_text": "some chunk text",
        },
    )


def _below_threshold_results() -> list[RetrievalResult]:
    return [_result("chunk-1", 1, THRESHOLD - 0.2)]


def _confident_results() -> list[RetrievalResult]:
    return [_result("chunk-1", 1, THRESHOLD + 0.2), _result("chunk-2", 2, THRESHOLD + 0.1)]


def _many_confident_results(n: int) -> list[RetrievalResult]:
    return [_result(f"chunk-{i}", i, THRESHOLD + 0.3 - i * 0.001) for i in range(1, n + 1)]


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_refused_by_threshold_never_calls_llm(mock_retrieve, mock_generate):
    mock_retrieve.return_value = _below_threshold_results()

    response = answer_question("What is the SOP?", "en", settings=_settings())

    mock_generate.assert_not_called()
    assert response.refused is True
    assert response.status == "ok"
    assert response.answer == REFUSAL_MESSAGE["en"]
    assert response.citations == []
    assert response.confidence == THRESHOLD - 0.2
    assert response.threshold == THRESHOLD


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_refused_by_threshold_spanish_uses_spanish_message(mock_retrieve, mock_generate):
    mock_retrieve.return_value = _below_threshold_results()

    response = answer_question("Cual es el SOP?", "es", settings=_settings())

    mock_generate.assert_not_called()
    assert response.refused is True
    assert response.answer == REFUSAL_MESSAGE["es"]
    assert response.language == "es"


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_normal_answer_path_resolves_citations_from_retrieved_metadata(mock_retrieve, mock_generate):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.return_value = {
        "answer": "The QC unit is responsible for X.",
        "citations": [{"chunk_id": "chunk-1"}],
        "refused": False,
    }

    response = answer_question("What is the QC unit responsible for?", "en", settings=_settings())

    assert response.refused is False
    assert response.status == "ok"
    assert response.answer == "The QC unit is responsible for X."
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.chunk_id == "chunk-1"
    assert citation.document_id == "doc-chunk-1"
    assert citation.document_title == "Title chunk-1"
    assert citation.section_heading == "Section chunk-1"
    assert citation.revision == "Rev A"
    assert response.confidence == THRESHOLD + 0.2
    assert response.threshold == THRESHOLD


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_llm_self_refusal_path_drops_citations_even_if_present(mock_retrieve, mock_generate):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.return_value = {
        "answer": "",
        "citations": [{"chunk_id": "chunk-1"}],
        "refused": True,
    }

    response = answer_question("An underspecified question", "en", settings=_settings())

    assert response.refused is True
    assert response.citations == []
    assert response.answer == REFUSAL_MESSAGE["en"]


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_llm_self_refusal_uses_llm_answer_text_when_non_empty(mock_retrieve, mock_generate):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.return_value = {
        "answer": "I cannot answer this confidently.",
        "citations": [{"chunk_id": "chunk-1"}],
        "refused": True,
    }

    response = answer_question("An underspecified question", "en", settings=_settings())

    assert response.refused is True
    assert response.citations == []
    assert response.answer == "I cannot answer this confidently."


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_generation_error_path_returns_error_status(mock_retrieve, mock_generate):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.side_effect = GenerationError("both providers failed")

    response = answer_question("What is the QC unit responsible for?", "en", settings=_settings())

    assert response.refused is False
    assert response.status == "error"
    assert response.answer == GENERATION_ERROR_MESSAGE["en"]
    assert response.citations == []
    assert response.confidence == THRESHOLD + 0.2
    assert response.threshold == THRESHOLD


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_generation_error_path_spanish_uses_spanish_message(mock_retrieve, mock_generate):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.side_effect = GenerationError("both providers failed")

    response = answer_question("Cual es la responsabilidad?", "es", settings=_settings())

    assert response.status == "error"
    assert response.answer == GENERATION_ERROR_MESSAGE["es"]
    assert response.language == "es"


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_request_id_present_and_looks_like_a_uuid(mock_retrieve, mock_generate):
    mock_retrieve.return_value = _below_threshold_results()

    response = answer_question("What is the SOP?", "en", settings=_settings())

    assert uuid.UUID(response.request_id)


@patch("api.generation._resolve_citations")
@patch("api.prompts.build_user_prompt")
@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_retrieve_called_with_wide_k_but_only_top_5_passed_downstream(
    mock_retrieve, mock_generate, mock_build_user_prompt, mock_resolve_citations
):
    results = _many_confident_results(8)
    mock_retrieve.return_value = results
    mock_build_user_prompt.return_value = "a built prompt"
    mock_generate.return_value = {
        "answer": "An answer.",
        "citations": [{"chunk_id": "chunk-1"}],
        "refused": False,
    }
    mock_resolve_citations.return_value = [
        Citation(
            document_id="doc-chunk-1",
            document_title="Title chunk-1",
            section_heading="Section chunk-1",
            revision="Rev A",
            chunk_id="chunk-1",
        )
    ]

    answer_question("What is the SOP?", "en", settings=_settings())

    mock_retrieve.assert_called_once_with("What is the SOP?", k=SEMANTIC_EXTRACTION_K)

    prompt_call = mock_build_user_prompt.call_args
    assert prompt_call.args[0] == "What is the SOP?"
    assert prompt_call.args[1] == results[:5]
    assert len(prompt_call.args[1]) == 5

    resolve_call = mock_resolve_citations.call_args
    assert resolve_call.args[1] == results[:5]
    assert len(resolve_call.args[1]) == 5


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_confident_non_refused_answer_with_empty_citations_downgrades_to_refusal(mock_retrieve, mock_generate):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.return_value = {
        "answer": "An answer with no citations at all.",
        "citations": [],
        "refused": False,
    }

    response = answer_question("What is the QC unit responsible for?", "en", settings=_settings())

    assert response.refused is True
    assert response.status == "ok"
    assert response.answer == REFUSAL_MESSAGE["en"]
    assert response.citations == []


@patch("api.llm_client.generate_structured")
@patch("retrieval.hybrid.retrieve")
def test_confident_non_refused_answer_with_only_unmatched_citations_downgrades_to_refusal(
    mock_retrieve, mock_generate
):
    results = _confident_results()
    mock_retrieve.return_value = results
    mock_generate.return_value = {
        "answer": "An answer citing a chunk that was never retrieved.",
        "citations": [{"chunk_id": "chunk-not-in-retrieved-set"}],
        "refused": False,
    }

    response = answer_question("What is the QC unit responsible for?", "en", settings=_settings())

    assert response.refused is True
    assert response.status == "ok"
    assert response.answer == REFUSAL_MESSAGE["en"]
    assert response.citations == []
