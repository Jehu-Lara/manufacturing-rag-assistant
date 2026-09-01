from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from src.core.config import Settings
from src.core.errors import GenerationError
from src.core.telemetry import get_tracer
from src.domain.models import DecisionReason, GateBand, Language, QueryAnswer
from src.domain.policies import CitationResolver, GroundedEvidenceResolver, RefusalPolicy
from src.domain.ports import LLMClientPort, RetrieverPort
from src.features.query.prompts import (
    GENERATION_ERROR_MESSAGE,
    GROUNDED_REVIEW_SCHEMA,
    JSON_SCHEMA,
    REFUSAL_MESSAGE,
    build_grounded_review_system_prompt,
    build_system_prompt,
    build_user_prompt,
)
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

logger = logging.getLogger(__name__)

PROMPT_CONTEXT_K = 5


def _log_query_completed(
    request_id: str,
    refused: bool,
    status: str,
    gate_band: GateBand,
    decision_reason: DecisionReason,
    confidence: Optional[float],
    start_time: float,
) -> None:
    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "query completed",
        extra={
            "request_id": request_id,
            "event": "query_completed",
            "refused": refused,
            "status": status,
            "gate_band": gate_band,
            "decision_reason": decision_reason,
            "confidence": confidence,
            "latency_ms": latency_ms,
        },
    )


class QueryUseCase:
    def __init__(self, retriever: RetrieverPort, llm_client: LLMClientPort, settings: Settings) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._settings = settings
        self._refusal_policy = RefusalPolicy(
            settings.refusal_cosine_threshold,
            mode=settings.refusal_policy,
            review_floor=settings.refusal_review_floor,
        )

    @property
    def _response_review_floor(self) -> Optional[float]:
        if self._settings.refusal_policy == "grounded_review":
            return self._settings.refusal_review_floor
        return None

    async def answer_question(self, question: str, language: Language) -> QueryAnswer:
        """Outer span correlating retrieval.hybrid.query/embedder.compute/
        llm.generate (see ADR-006) under one trace — asyncio.to_thread copies
        the current contextvars context to its worker thread (per the
        stdlib docs), so the retriever's child span still nests correctly
        even though retrieve() runs off the event loop."""
        with get_tracer().start_as_current_span("query.answer_question"):
            return await self._answer_question_impl(question, language)

    def _answer(
        self,
        *,
        answer: str,
        citations: list[Any],
        refused: bool,
        status: str,
        confidence: Optional[float],
        gate_band: GateBand,
        decision_reason: DecisionReason,
        language: Language,
        request_id: str,
    ) -> QueryAnswer:
        return QueryAnswer(
            answer=answer,
            citations=citations,
            refused=refused,
            status=status,  # type: ignore[arg-type]
            confidence=confidence,
            threshold=self._settings.refusal_cosine_threshold,
            review_floor=self._response_review_floor,
            gate_band=gate_band,
            decision_reason=decision_reason,
            language=language,
            request_id=request_id,
        )

    def _refusal_answer(
        self,
        *,
        language: Language,
        confidence: Optional[float],
        gate_band: GateBand,
        decision_reason: DecisionReason,
        request_id: str,
        answer_text: Optional[str] = None,
    ) -> QueryAnswer:
        return self._answer(
            answer=answer_text or REFUSAL_MESSAGE[language],
            citations=[],
            refused=True,
            status="ok",
            confidence=confidence,
            gate_band=gate_band,
            decision_reason=decision_reason,
            language=language,
            request_id=request_id,
        )

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

        score = self._refusal_policy.top1_semantic_score(results)
        gate_band = self._refusal_policy.classify_score(score)
        prompt_results = results[:PROMPT_CONTEXT_K]

        if gate_band == "hard_refuse":
            decision_reason = self._refusal_policy.hard_refuse_reason()
            answer = self._refusal_answer(
                language=language,
                confidence=score,
                gate_band=gate_band,
                decision_reason=decision_reason,
                request_id=request_id,
            )
            _log_query_completed(
                request_id, True, "ok", gate_band, decision_reason, score, start_time
            )
            return answer

        if gate_band == "grounded_review":
            system_prompt = build_grounded_review_system_prompt(language)
            schema = GROUNDED_REVIEW_SCHEMA
        else:
            system_prompt = build_system_prompt(language)
            schema = JSON_SCHEMA
        user_prompt = build_user_prompt(question, prompt_results)

        # One logical generation call. The adapter underneath may still perform
        # physical provider retries, JSON repair, or a primary->fallback provider
        # switch — "one call" is the gate's contract, not a network guarantee.
        try:
            llm_result = await self._llm_client.generate_structured(
                system_prompt, user_prompt, schema, self._settings
            )
        except GenerationError as exc:
            logger.error(
                "structured generation failed",
                extra={
                    "request_id": request_id,
                    "event": "generation_error",
                    "language": language,
                    "gate_band": gate_band,
                    "error_type": type(exc).__name__,
                },
            )
            answer = self._answer(
                answer=GENERATION_ERROR_MESSAGE[language],
                citations=[],
                refused=False,
                status="error",
                confidence=score,
                gate_band=gate_band,
                decision_reason="generation_error",
                language=language,
                request_id=request_id,
            )
            _log_query_completed(
                request_id, False, "error", gate_band, "generation_error", score, start_time
            )
            return answer

        if bool(llm_result.get("refused", False)):
            # Grey band: the model is instructed to echo the verbatim refusal
            # string, but at borderline confidence arbitrary self-refusal text is
            # not trusted — always return the canonical message. binary/confident
            # keeps the legacy behaviour of passing the model's refusal text through.
            if gate_band == "grounded_review":
                answer_text = None
            else:
                answer_text = llm_result.get("answer") or None
            answer = self._refusal_answer(
                language=language,
                confidence=score,
                gate_band=gate_band,
                decision_reason="llm_self_refusal",
                request_id=request_id,
                answer_text=answer_text,
            )
            _log_query_completed(
                request_id, True, "ok", gate_band, "llm_self_refusal", score, start_time
            )
            return answer

        # Applies to every band, including binary/confident (ADR-009 §2): a blank
        # non-refused answer is never useful, so degrade it to the canonical
        # refusal rather than serve an empty body.
        if not str(llm_result.get("answer") or "").strip():
            logger.info(
                "non-refused answer body was empty or whitespace; refusing",
                extra={
                    "request_id": request_id,
                    "event": "empty_answer_downgraded_to_refusal",
                    "language": language,
                    "gate_band": gate_band,
                },
            )
            answer = self._refusal_answer(
                language=language,
                confidence=score,
                gate_band=gate_band,
                decision_reason="empty_answer",
                request_id=request_id,
            )
            _log_query_completed(
                request_id, True, "ok", gate_band, "empty_answer", score, start_time
            )
            return answer

        if gate_band == "grounded_review":
            grounding = GroundedEvidenceResolver.resolve(
                llm_result.get("evidence", []), prompt_results
            )
            if grounding.failure_reason is not None:
                logger.info(
                    "grounded-review answer failed evidence validation; refusing",
                    extra={
                        "request_id": request_id,
                        "event": "grounded_review_downgraded",
                        "language": language,
                        "decision_reason": grounding.failure_reason,
                    },
                )
                answer = self._refusal_answer(
                    language=language,
                    confidence=score,
                    gate_band=gate_band,
                    decision_reason=grounding.failure_reason,
                    request_id=request_id,
                )
                _log_query_completed(
                    request_id, True, "ok", gate_band, grounding.failure_reason, score, start_time
                )
                return answer
            citations = grounding.citations
            decision_reason = "accepted_grounded"
        else:
            citations = CitationResolver.resolve(llm_result.get("citations", []), prompt_results)
            if not citations:
                logger.info(
                    "confident, non-refused answer had no resolvable citations; downgrading to refusal",
                    extra={
                        "request_id": request_id,
                        "event": "uncited_answer_downgraded_to_refusal",
                        "language": language,
                    },
                )
                answer = self._refusal_answer(
                    language=language,
                    confidence=score,
                    gate_band=gate_band,
                    decision_reason="unresolved_citation",
                    request_id=request_id,
                )
                _log_query_completed(
                    request_id, True, "ok", gate_band, "unresolved_citation", score, start_time
                )
                return answer
            decision_reason = "accepted_confident"

        answer = self._answer(
            answer=str(llm_result.get("answer") or ""),
            citations=citations,
            refused=False,
            status="ok",
            confidence=score,
            gate_band=gate_band,
            decision_reason=decision_reason,
            language=language,
            request_id=request_id,
        )
        _log_query_completed(
            request_id, False, "ok", gate_band, decision_reason, score, start_time
        )
        return answer
