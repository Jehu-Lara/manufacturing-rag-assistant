from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter, get_settings, get_vector_store
from src.adapters.primary.http.rate_limit import RateLimiter
from src.adapters.primary.http.schemas import Citation, HealthResponse, QueryRequest, QueryResponse, ReadyResponse
from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME
from src.core.config import Settings
from src.domain.models import QueryAnswer
from src.domain.ports import VectorStorePort
from src.features.query.use_cases import QueryUseCase

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_response_schema(answer: QueryAnswer) -> QueryResponse:
    return QueryResponse(
        answer=answer.answer,
        citations=[
            Citation(
                document_id=c.document_id,
                document_title=c.document_title,
                section_heading=c.section_heading,
                revision=c.revision,
                chunk_id=c.chunk_id,
            )
            for c in answer.citations
        ],
        refused=answer.refused,
        status=answer.status,
        confidence=answer.confidence,
        threshold=answer.threshold,
        language=answer.language,
        request_id=answer.request_id,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    vector_store: VectorStorePort = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    index_loaded = vector_store.ping()
    if not index_loaded:
        logger.warning(
            "vector store collection unreachable during health check",
            extra={"event": "health_check_index_unreachable"},
        )
    return HealthResponse(
        status="ok",
        embedding_model=MODEL_NAME,
        llm_provider_primary=settings.llm_provider,
        index_loaded=index_loaded,
    )


@router.get("/ready")
async def ready(vector_store: VectorStorePort = Depends(get_vector_store)) -> JSONResponse:
    """Additive, distinct from /health (always 200, liveness-only, contract
    unchanged): returns 503 when the vector store isn't actually loaded and
    queryable, so a readiness probe (Cloud Run/K8s/HF Spaces) can avoid
    routing traffic to a container whose index never loaded."""
    if vector_store.ping():
        return JSONResponse(status_code=200, content=ReadyResponse(status="ready").model_dump())
    return JSONResponse(status_code=503, content=ReadyResponse(status="not_ready").model_dump())


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    http_request: Request,
    query_use_case: QueryUseCase = Depends(get_query_use_case),
    settings: Settings = Depends(get_settings),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> QueryResponse:
    expected_api_key = settings.api_key.get_secret_value() if settings.api_key is not None else None
    if expected_api_key is not None and not secrets.compare_digest(x_api_key or "", expected_api_key):
        logger.warning("rejected request with missing or invalid API key", extra={"event": "invalid_api_key"})
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    client_key = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.allow(client_key):
        logger.warning("rate limit exceeded", extra={"event": "rate_limit_exceeded", "client": client_key})
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    answer = await query_use_case.answer_question(request.question, request.language)
    return _to_response_schema(answer)
