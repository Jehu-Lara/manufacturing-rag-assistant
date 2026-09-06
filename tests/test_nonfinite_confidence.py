from __future__ import annotations

import asyncio

import pytest

from src.core.config import RefusalPolicyName, Settings
from src.domain.models import GateBand, Language, RetrievalResult
from src.domain.policies import RefusalPolicy, is_confident
from src.features.query.prompts import REFUSAL_MESSAGE
from src.features.query.use_cases import QueryUseCase
from tests.fakes import InMemoryLLMClient, InMemoryRetriever

THRESHOLD = 0.5999
FLOOR = 0.5500
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def _result(score: float | None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="fixture::chunk-0000",
        fused_score=1.0 / 61,
        semantic_rank=1,
        semantic_score=score,
        bm25_rank=None,
        bm25_score=None,
        metadata={
            "document_id": "fixture",
            "document_title": "Fixture",
            "section_heading": "Procedure",
            "revision": "Rev A",
            "source_type": "synthetic",
            "chunk_text": "This complete fixture context must not reach generation.",
        },
    )


@pytest.mark.parametrize("mode", ["binary", "grounded_review"])
@pytest.mark.parametrize("score", NON_FINITE)
def test_nonfinite_policy_confidence_fails_closed(mode: RefusalPolicyName, score: float) -> None:
    policy = RefusalPolicy(THRESHOLD, mode=mode, review_floor=FLOOR)
    results = [_result(score)]

    assert policy.classify_score(score) == "hard_refuse"
    assert policy.classify(results) == "hard_refuse"
    assert policy.top1_semantic_score(results) is None


@pytest.mark.parametrize("score", NON_FINITE)
def test_nonfinite_scalar_confidence_is_false(score: float) -> None:
    assert is_confident(score, THRESHOLD) is False


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("mode", ["binary", "grounded_review"])
@pytest.mark.parametrize("score", NON_FINITE)
def test_nonfinite_query_refuses_without_generation(
    language: Language, mode: RefusalPolicyName, score: float
) -> None:
    settings = Settings(
        groq_api_key="groq-test-key",
        openai_api_key="openai-test-key",
        llm_provider="groq",
        refusal_cosine_threshold=THRESHOLD,
        refusal_policy=mode,
        refusal_review_floor=FLOOR,
        log_level="INFO",
    )
    llm = InMemoryLLMClient(error=AssertionError("generation must not be called"))
    use_case = QueryUseCase(InMemoryRetriever([_result(score)]), llm, settings)

    answer = asyncio.run(use_case.answer_question("Fixture question", language))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.gate_band == "hard_refuse"
    assert answer.confidence is None
    assert answer.answer == REFUSAL_MESSAGE[language]
    assert answer.citations == []
    assert answer.threshold == THRESHOLD
    assert answer.review_floor == (FLOOR if mode == "grounded_review" else None)


@pytest.mark.parametrize(
    "score,binary_band,review_band",
    [
        (None, "hard_refuse", "hard_refuse"),
        (-0.1, "hard_refuse", "hard_refuse"),
        (0.5500, "hard_refuse", "grounded_review"),
        (0.5998, "hard_refuse", "grounded_review"),
        (0.5999, "confident", "confident"),
        (0.7, "confident", "confident"),
    ],
)
def test_finite_and_missing_confidence_keep_existing_behavior(
    score: float | None, binary_band: GateBand, review_band: GateBand
) -> None:
    result = _result(score)
    binary = RefusalPolicy(THRESHOLD)
    review = RefusalPolicy(THRESHOLD, mode="grounded_review", review_floor=FLOOR)

    assert binary.classify([result]) == binary_band
    assert review.classify([result]) == review_band
    assert binary.top1_semantic_score([result]) == score
    assert review.top1_semantic_score([result]) == score
    assert is_confident(score, THRESHOLD) == (binary_band == "confident")
    assert result.semantic_score == score
