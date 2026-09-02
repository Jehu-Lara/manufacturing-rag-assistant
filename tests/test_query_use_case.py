from __future__ import annotations

import asyncio
import logging
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


GROUNDED_THRESHOLD = 0.5999
GROUNDED_FLOOR = 0.5500
_GREY_SCORE = 0.5700
_GREY_CHUNK_TEXT = (
    "Net positive suction head available (NPSHA) is a property of the system; net positive "
    "suction head required (NPSHR) is a property of the pump and is set by the manufacturer."
)


def _grounded_settings() -> Settings:
    return Settings(
        groq_api_key="groq-test-key",
        openai_api_key="openai-test-key",
        llm_provider="groq",
        refusal_cosine_threshold=GROUNDED_THRESHOLD,
        refusal_policy="grounded_review",
        refusal_review_floor=GROUNDED_FLOOR,
        log_level="INFO",
    )


def _grey_results() -> list[RetrievalResult]:
    return [_result("chunk-1", 1, _GREY_SCORE, chunk_text=_GREY_CHUNK_TEXT)]


def _result(
    chunk_id: str, semantic_rank: int, semantic_score: float, chunk_text: str = "some chunk text"
) -> RetrievalResult:
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
            "source_type": "public",
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
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


# --- Phase 3: binary policy carries the new fields but keeps behaviour ---


def test_binary_default_answer_carries_new_fields():
    llm = InMemoryLLMClient(
        response={"answer": "A.", "citations": [{"chunk_id": "chunk-1"}], "refused": False}
    )
    answer = _run(
        QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings()).answer_question(
            "q", "en"
        )
    )
    assert answer.gate_band == "confident"
    assert answer.decision_reason == "accepted_confident"
    assert answer.review_floor is None  # binary policy -> always null


def test_binary_threshold_refusal_reason_and_band():
    answer = _run(
        QueryUseCase(
            InMemoryRetriever(_below_threshold_results()), InMemoryLLMClient(), _settings()
        ).answer_question("q", "en")
    )
    assert answer.gate_band == "hard_refuse"
    assert answer.decision_reason == "below_binary_threshold"


# --- Phase 3: grounded_review policy ---


class _CountingLLM(InMemoryLLMClient):
    def __init__(self, response=None, error=None) -> None:
        super().__init__(response=response, error=error)
        self.schemas_seen: list[dict] = []
        self.calls = 0

    async def generate_structured(self, system_prompt, user_prompt, schema, settings):
        self.calls += 1
        self.schemas_seen.append(schema)
        return await super().generate_structured(system_prompt, user_prompt, schema, settings)


def test_grounded_hard_refuse_below_floor_never_calls_llm():
    llm = _CountingLLM()
    results = [_result("chunk-1", 1, 0.40, chunk_text=_GREY_CHUNK_TEXT)]
    answer = _run(
        QueryUseCase(InMemoryRetriever(results), llm, _grounded_settings()).answer_question("q", "en")
    )
    assert llm.calls == 0
    assert answer.refused is True
    assert answer.gate_band == "hard_refuse"
    assert answer.decision_reason == "below_review_floor"
    assert answer.review_floor == GROUNDED_FLOOR


def test_grounded_review_accepts_verbatim_evidence_with_one_llm_call():
    quote = "net positive suction head required (NPSHR) is a property of the pump"
    llm = _CountingLLM(
        response={
            "answer": "NPSHA is a system property; NPSHR is a pump property.",
            "evidence": [{"chunk_id": "chunk-1", "supporting_quote": quote}],
            "refused": False,
        }
    )
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "What is the difference?", "en"
        )
    )
    assert llm.calls == 1
    assert "evidence" in llm.schemas_seen[0]["properties"]
    assert answer.refused is False
    assert answer.gate_band == "grounded_review"
    assert answer.decision_reason == "accepted_grounded"
    assert answer.review_floor == GROUNDED_FLOOR
    assert [c.chunk_id for c in answer.citations] == ["chunk-1"]


def test_grounded_review_downgrades_on_fabricated_quote():
    llm = _CountingLLM(
        response={
            "answer": "Fabricated.",
            "evidence": [
                {"chunk_id": "chunk-1", "supporting_quote": "a quote that is not in the chunk at all here"}
            ],
            "refused": False,
        }
    )
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "q", "en"
        )
    )
    assert answer.refused is True
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.decision_reason == "quote_not_found"
    assert answer.citations == []


def test_grounded_review_empty_evidence_downgrades():
    llm = _CountingLLM(
        response={"answer": "Unsupported.", "evidence": [], "refused": False}
    )
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "q", "es"
        )
    )
    assert answer.refused is True
    assert answer.answer == REFUSAL_MESSAGE["es"]
    assert answer.decision_reason == "missing_evidence"


def test_grounded_review_llm_self_refusal():
    llm = _CountingLLM(response={"answer": "", "evidence": [], "refused": True})
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "q", "en"
        )
    )
    assert answer.refused is True
    assert answer.decision_reason == "llm_self_refusal"
    assert answer.answer == REFUSAL_MESSAGE["en"]


def test_grounded_review_provider_error_keeps_error_status():
    llm = _CountingLLM(error=GenerationError("both providers failed"))
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "q", "en"
        )
    )
    assert answer.status == "error"
    assert answer.refused is False
    assert answer.decision_reason == "generation_error"
    assert answer.gate_band == "grounded_review"


def test_grounded_review_self_refusal_ignores_non_canonical_answer_text():
    llm = _CountingLLM(
        response={
            "answer": "Actually here is a partial answer I am not sure about.",
            "evidence": [],
            "refused": True,
        }
    )
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "q", "en"
        )
    )
    assert answer.refused is True
    assert answer.decision_reason == "llm_self_refusal"
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.citations == []


def test_grounded_review_empty_answer_with_valid_evidence_downgrades():
    quote = "net positive suction head required (NPSHR) is a property of the pump"
    llm = _CountingLLM(
        response={
            "answer": "   \n\t ",
            "evidence": [{"chunk_id": "chunk-1", "supporting_quote": quote}],
            "refused": False,
        }
    )
    answer = _run(
        QueryUseCase(InMemoryRetriever(_grey_results()), llm, _grounded_settings()).answer_question(
            "q", "en"
        )
    )
    assert answer.refused is True
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.decision_reason == "empty_answer"
    assert answer.citations == []


def test_confident_empty_answer_with_valid_citations_downgrades():
    llm = InMemoryLLMClient(
        response={"answer": "", "citations": [{"chunk_id": "chunk-1"}], "refused": False}
    )
    use_case = QueryUseCase(InMemoryRetriever(_confident_results()), llm, _settings())

    answer = _run(use_case.answer_question("q", "en"))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.decision_reason == "empty_answer"
    assert answer.citations == []


def test_grounded_review_never_logs_question_answer_or_quote(caplog):
    secret_question = "WHAT-IS-NPSHA-SENTINEL-QUESTION"
    secret_answer = "SENTINEL-ANSWER-BODY-THAT-MUST-NOT-BE-LOGGED"
    quote = "net positive suction head required (NPSHR) is a property of the pump"
    llm = _CountingLLM(
        response={
            "answer": secret_answer,
            "evidence": [{"chunk_id": "chunk-1", "supporting_quote": quote}],
            "refused": False,
        }
    )
    with caplog.at_level(logging.DEBUG):
        _run(
            QueryUseCase(
                InMemoryRetriever(_grey_results()), llm, _grounded_settings()
            ).answer_question(secret_question, "en")
        )

    for record in caplog.records:
        blob = " ".join(
            [record.getMessage(), *(str(v) for v in record.__dict__.values())]
        )
        assert secret_question not in blob
        assert secret_answer not in blob
        assert quote not in blob


def test_grounded_confident_band_uses_normal_schema():
    llm = _CountingLLM(
        response={"answer": "A.", "citations": [{"chunk_id": "chunk-1"}], "refused": False}
    )
    results = [_result("chunk-1", 1, 0.80, chunk_text=_GREY_CHUNK_TEXT)]
    answer = _run(
        QueryUseCase(InMemoryRetriever(results), llm, _grounded_settings()).answer_question("q", "en")
    )
    assert answer.gate_band == "confident"
    assert answer.decision_reason == "accepted_confident"
    assert "citations" in llm.schemas_seen[0]["properties"]
