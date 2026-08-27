from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import api.generation
import api.logging_setup
import retrieval.vector_store
from api.config import Settings, load_settings
from api.rate_limit import RateLimiter
from api.schemas import HealthResponse, QueryRequest, QueryResponse
from retrieval.embedder import MODEL_NAME

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    api.logging_setup.configure(level=settings.log_level)
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter(max_requests=settings.rate_limit_per_minute)
    yield


app = FastAPI(title="Manufacturing Knowledge RAG Assistant", lifespan=lifespan)

# No cookies/credentials are used anywhere in this API, so an open origin
# list carries no CSRF/credential-theft risk here — this only stops browsers
# from silently dropping cross-origin responses for a legitimate third-party
# caller (e.g. the showcase page), not a security boundary in itself; the
# actual access control is the optional API-key check on /query below.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception",
        exc_info=exc,
        extra={"event": "unhandled_exception", "path": request.url.path},
    )
    return JSONResponse(status_code=500, content={"status": "error", "detail": "internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        retrieval.vector_store.get_collection()
        index_loaded = True
    except Exception as exc:
        logger.warning(
            "vector store collection unreachable during health check",
            extra={"event": "health_check_index_unreachable", "error": str(exc)},
        )
        index_loaded = False

    settings: Settings = app.state.settings
    return HealthResponse(
        status="ok",
        embedding_model=MODEL_NAME,
        llm_provider_primary=settings.llm_provider,
        index_loaded=index_loaded,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    http_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> QueryResponse:
    settings: Settings = http_request.app.state.settings
    if settings.api_key is not None and x_api_key != settings.api_key:
        logger.warning(
            "rejected request with missing or invalid API key",
            extra={"event": "invalid_api_key"},
        )
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    rate_limiter: RateLimiter = http_request.app.state.rate_limiter
    client_key = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.allow(client_key):
        logger.warning(
            "rate limit exceeded",
            extra={"event": "rate_limit_exceeded", "client": client_key},
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    return api.generation.answer_question(request.question, request.language, settings)
