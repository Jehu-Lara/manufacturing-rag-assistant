from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import api.llm_client
import api.messages
import api.prompts
import api.refusal
import retrieval.hybrid
from api.config import Settings, load_settings
from api.schemas import Citation, QueryResponse

logger = logging.getLogger(__name__)


def _resolve_citations(llm_citations: list[dict], results: list) -> list[Citation]:
    results_by_chunk_id = {result.chunk_id: result for result in results}
    resolved: list[Citation] = []
    for llm_citation in llm_citations:
        chunk_id = llm_citation.get("chunk_id")
        result = results_by_chunk_id.get(chunk_id)
        if result is None:
            logger.warning(
                "LLM cited a chunk_id not among the retrieved chunks; dropping citation",
                extra={"event": "citation_not_in_retrieved_set", "chunk_id": chunk_id},
            )
            continue
        metadata = result.metadata
        resolved.append(
            Citation(
                document_id=metadata["document_id"],
                document_title=metadata["document_title"],
                section_heading=metadata["section_heading"],
                revision=metadata["revision"],
                chunk_id=result.chunk_id,
            )
        )
    return resolved


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


def answer_question(question: str, language: str, settings: Optional[Settings] = None) -> QueryResponse:
    if settings is None:
        settings = load_settings()

    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    results = retrieval.hybrid.retrieve(question, k=5)
    logger.info(
        "query received",
        extra={"request_id": request_id, "event": "query_received", "language": language},
    )

    top1_semantic_score = api.refusal.top1_semantic_score_from_results(results)
    confident = api.refusal.is_confident(top1_semantic_score, settings.refusal_cosine_threshold)

    if not confident:
        logger.info(
            "query refused: retrieval confidence below threshold",
            extra={
                "request_id": request_id,
                "event": "query_refused",
                "language": language,
                "confidence": top1_semantic_score,
                "threshold": settings.refusal_cosine_threshold,
            },
        )
        response = QueryResponse(
            answer=api.messages.REFUSAL_MESSAGE[language],
            citations=[],
            refused=True,
            status="ok",
            confidence=top1_semantic_score,
            threshold=settings.refusal_cosine_threshold,
            language=language,
            request_id=request_id,
        )
        _log_query_completed(request_id, response.refused, response.status, start_time)
        return response

    system_prompt = api.prompts.build_system_prompt(language)
    user_prompt = api.prompts.build_user_prompt(question, results)

    try:
        llm_result = api.llm_client.generate_structured(system_prompt, user_prompt, api.prompts.JSON_SCHEMA, settings)
    except api.llm_client.GenerationError as exc:
        logger.error(
            "structured generation failed",
            extra={
                "request_id": request_id,
                "event": "generation_error",
                "language": language,
                "error": str(exc),
            },
        )
        response = QueryResponse(
            answer=api.messages.GENERATION_ERROR_MESSAGE[language],
            citations=[],
            refused=False,
            status="error",
            confidence=top1_semantic_score,
            threshold=settings.refusal_cosine_threshold,
            language=language,
            request_id=request_id,
        )
        _log_query_completed(request_id, response.refused, response.status, start_time)
        return response

    llm_refused = bool(llm_result.get("refused", False))
    llm_answer = llm_result.get("answer") or ""

    if llm_refused:
        response = QueryResponse(
            answer=llm_answer if llm_answer else api.messages.REFUSAL_MESSAGE[language],
            citations=[],
            refused=True,
            status="ok",
            confidence=top1_semantic_score,
            threshold=settings.refusal_cosine_threshold,
            language=language,
            request_id=request_id,
        )
    else:
        llm_citations = llm_result.get("citations", [])
        citations = _resolve_citations(llm_citations, results)
        response = QueryResponse(
            answer=llm_answer,
            citations=citations,
            refused=False,
            status="ok",
            confidence=top1_semantic_score,
            threshold=settings.refusal_cosine_threshold,
            language=language,
            request_id=request_id,
        )

    _log_query_completed(request_id, response.refused, response.status, start_time)
    return response
