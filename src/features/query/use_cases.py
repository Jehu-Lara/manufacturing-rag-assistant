from __future__ import annotations

import asyncio
import logging
import time
import uuid

from src.core.config import Settings
from src.core.errors import GenerationError
from src.core.telemetry import get_tracer
from src.domain.models import Language, QueryAnswer
from src.domain.policies import CitationResolver, RefusalPolicy
from src.domain.ports import LLMClientPort, RetrieverPort
from src.features.query.prompts import (
    GENERATION_ERROR_MESSAGE,
    JSON_SCHEMA,
    REFUSAL_MESSAGE,
    build_system_prompt,
    build_user_prompt,
)
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

logger = logging.getLogger(__name__)

PROMPT_CONTEXT_K = 5


def _log_query_completed(request_id: str, refused: bool, status: str, start_time: float) -> None:
    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "query completed",
        extra={
            "request_id": request_id,
            "event": "query_completed",
            "refused": refused,
            "status": status,
            "latency_ms": latency_ms,
        },
    )


class QueryUseCase:
    def __init__(self, retriever: RetrieverPort, llm_client: LLMClientPort, settings: Settings) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._settings = settings
        self._refusal_policy = RefusalPolicy(settings.refusal_cosine_threshold)

    async def answer_question(self, question: str, language: Language) -> QueryAnswer:
        """Outer span correlating retrieval.hybrid.query/embedder.compute/
        llm.generate (see ADR-006) under one trace — asyncio.to_thread copies
        the current contextvars context to its worker thread (per the
        stdlib docs), so the retriever's child span still nests correctly
        even though retrieve() runs off the event loop."""
        with get_tracer().start_as_current_span("query.answer_question"):
            return await self._answer_question_impl(question, language)

    async def _answer_question_impl(self, question: str, language: Language) -> QueryAnswer:
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # The retriever (sentence-transformers encode + Chroma query + BM25
        # search) is synchronous — no async API in the pinned versions — but
        # must not block the event loop from inside this async method.
        results = await asyncio.to_thread(self._retriever.retrieve, question, k=SEMANTIC_EXTRACTION_K)
        logger.info(
            "query received",
            extra={"request_id": request_id, "event": "query_received", "language": language},
        )

        top1_semantic_score = self._refusal_policy.top1_semantic_score(results)
        confident = self._refusal_policy.is_confident(results)
        prompt_results = results[:PROMPT_CONTEXT_K]

        if not confident:
            logger.info(
                "query refused: retrieval confidence below threshold",
                extra={
                    "request_id": request_id,
                    "event": "query_refused",
                    "language": language,
                    "confidence": top1_semantic_score,
                    "threshold": self._settings.refusal_cosine_threshold,
                },
            )
            answer = QueryAnswer(
                answer=REFUSAL_MESSAGE[language],
                citations=[],
                refused=True,
                status="ok",
                confidence=top1_semantic_score,
                threshold=self._settings.refusal_cosine_threshold,
                language=language,
                request_id=request_id,
            )
            _log_query_completed(request_id, answer.refused, answer.status, start_time)
            return answer

        system_prompt = build_system_prompt(language)
        user_prompt = build_user_prompt(question, prompt_results)

        try:
            llm_result = await self._llm_client.generate_structured(
                system_prompt, user_prompt, JSON_SCHEMA, self._settings
            )
        except GenerationError as exc:
            logger.error(
                "structured generation failed",
                extra={
                    "request_id": request_id,
                    "event": "generation_error",
                    "language": language,
                    "error": str(exc),
                },
            )
            answer = QueryAnswer(
                answer=GENERATION_ERROR_MESSAGE[language],
                citations=[],
                refused=False,
                status="error",
                confidence=top1_semantic_score,
                threshold=self._settings.refusal_cosine_threshold,
                language=language,
                request_id=request_id,
            )
            _log_query_completed(request_id, answer.refused, answer.status, start_time)
            return answer

        llm_refused = bool(llm_result.get("refused", False))
        llm_answer = llm_result.get("answer") or ""

        if llm_refused:
            answer = QueryAnswer(
                answer=llm_answer if llm_answer else REFUSAL_MESSAGE[language],
                citations=[],
                refused=True,
                status="ok",
                confidence=top1_semantic_score,
                threshold=self._settings.refusal_cosine_threshold,
                language=language,
                request_id=request_id,
            )
        else:
            llm_citations = llm_result.get("citations", [])
            citations = CitationResolver.resolve(llm_citations, prompt_results)
            if not citations:
                logger.info(
                    "confident, non-refused answer had no resolvable citations; downgrading to refusal",
                    extra={
                        "request_id": request_id,
                        "event": "uncited_answer_downgraded_to_refusal",
                        "language": language,
                    },
                )
                answer = QueryAnswer(
                    answer=REFUSAL_MESSAGE[language],
                    citations=[],
                    refused=True,
                    status="ok",
                    confidence=top1_semantic_score,
                    threshold=self._settings.refusal_cosine_threshold,
                    language=language,
                    request_id=request_id,
                )
            else:
                answer = QueryAnswer(
                    answer=llm_answer,
                    citations=citations,
                    refused=False,
                    status="ok",
                    confidence=top1_semantic_score,
                    threshold=self._settings.refusal_cosine_threshold,
                    language=language,
                    request_id=request_id,
                )

        _log_query_completed(request_id, answer.refused, answer.status, start_time)
        return answer
