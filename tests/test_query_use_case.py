from __future__ import annotations

import asyncio
import uuid

from src.core.config import Settings
from src.core.errors import GenerationError
from src.domain.models import RetrievalResult
from src.features.query.prompts import GENERATION_ERROR_MESSAGE, REFUSAL_MESSAGE
from src.features.query.use_cases import QueryUseCase
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K
from tests.fakes import InMemoryLLMClient, InMemoryRetriever

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


def _run(coro):
    return asyncio.run(coro)


class _SpyRetriever(InMemoryRetriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        super().__init__(results)
        self.calls: list[tuple] = []

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        self.calls.append((query_text, k))
        return super().retrieve(query_text, k, top_n)


def test_refused_by_threshold_never_calls_llm():
    llm = InMemoryLLMClient()
    use_case = QueryUseCase(InMemoryRetriever(_below_threshold_results()), llm, _settings())

    answer = _run(use_case.answer_question("What is the SOP?", "en"))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.citations == []
    assert answer.confidence == THRESHOLD - 0.2
    assert answer.threshold == THRESHOLD


def test_refused_by_threshold_spanish_uses_spanish_message():
    use_case = QueryUseCase(InMemoryRetriever(_below_threshold_results()), InMemoryLLMClient(), _settings())

    answer = _run(use_case.answer_question("Cual es el SOP?", "es"))

    assert answer.refused is True
    assert answer.answer == REFUSAL_MESSAGE["es"]
    assert answer.language == "es"


def test_normal_answer_path_resolves_citations_from_retrieved_metadata():
    llm = InMemoryLLMClient(
        response={
            "answer": "The QC unit is responsible for X.",
            "citations": [{"chunk_id": "chunk-1"}],
            "refused": False,
        }
    )
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("What is the QC unit responsible for?", "en"))

    assert answer.refused is False
    assert answer.status == "ok"
    assert answer.answer == "The QC unit is responsible for X."
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.chunk_id == "chunk-1"
    assert citation.document_id == "doc-chunk-1"
    assert citation.document_title == "Title chunk-1"
    assert citation.section_heading == "Section chunk-1"
    assert citation.revision == "Rev A"
    assert answer.confidence == THRESHOLD + 0.2
    assert answer.threshold == THRESHOLD


def test_llm_self_refusal_path_drops_citations_even_if_present():
    llm = InMemoryLLMClient(response={"answer": "", "citations": [{"chunk_id": "chunk-1"}], "refused": True})
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("An underspecified question", "en"))

    assert answer.refused is True
    assert answer.citations == []
    assert answer.answer == REFUSAL_MESSAGE["en"]


def test_llm_self_refusal_uses_llm_answer_text_when_non_empty():
    llm = InMemoryLLMClient(
        response={
            "answer": "I cannot answer this confidently.",
            "citations": [{"chunk_id": "chunk-1"}],
            "refused": True,
        }
    )
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("An underspecified question", "en"))

    assert answer.refused is True
    assert answer.citations == []
    assert answer.answer == "I cannot answer this confidently."


def test_generation_error_path_returns_error_status():
    llm = InMemoryLLMClient(error=GenerationError("both providers failed"))
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("What is the QC unit responsible for?", "en"))

    assert answer.refused is False
    assert answer.status == "error"
    assert answer.answer == GENERATION_ERROR_MESSAGE["en"]
    assert answer.citations == []
    assert answer.confidence == THRESHOLD + 0.2
    assert answer.threshold == THRESHOLD


def test_generation_error_path_spanish_uses_spanish_message():
    llm = InMemoryLLMClient(error=GenerationError("both providers failed"))
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("Cual es la responsabilidad?", "es"))

    assert answer.status == "error"
    assert answer.answer == GENERATION_ERROR_MESSAGE["es"]
    assert answer.language == "es"


def test_request_id_present_and_looks_like_a_uuid():
    use_case = QueryUseCase(InMemoryRetriever(_below_threshold_results()), InMemoryLLMClient(), _settings())

    answer = _run(use_case.answer_question("What is the SOP?", "en"))

    assert uuid.UUID(answer.request_id)


def test_retrieve_called_with_wide_k_but_only_top_5_passed_downstream():
    results = _many_confident_results(8)
    spy_retriever = _SpyRetriever(results)
    llm = InMemoryLLMClient(response={"answer": "An answer.", "citations": [{"chunk_id": "chunk-1"}], "refused": False})
    use_case = QueryUseCase(spy_retriever, llm, _settings())

    _run(use_case.answer_question("What is the SOP?", "en"))

    assert spy_retriever.calls == [("What is the SOP?", SEMANTIC_EXTRACTION_K)]


def test_confident_non_refused_answer_with_empty_citations_downgrades_to_refusal():
    llm = InMemoryLLMClient(
        response={"answer": "An answer with no citations at all.", "citations": [], "refused": False}
    )
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("What is the QC unit responsible for?", "en"))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.citations == []


def test_confident_non_refused_answer_with_only_unmatched_citations_downgrades_to_refusal():
    llm = InMemoryLLMClient(
        response={
            "answer": "An answer citing a chunk that was never retrieved.",
            "citations": [{"chunk_id": "chunk-not-in-retrieved-set"}],
            "refused": False,
        }
    )
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("What is the QC unit responsible for?", "en"))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.citations == []
