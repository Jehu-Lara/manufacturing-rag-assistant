from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.primary.http.rate_limit import RateLimiter
from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.llm.groq_openai_client import GroqOpenAiLlmClient
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.core.logging import configure as configure_logging
from src.core.telemetry import configure_tracing
from src.features.query.router import router
from src.features.query.use_cases import QueryUseCase
from src.features.retrieval.use_cases import HybridRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """All real adapter construction (model load, Chroma connection, BM25
    load) happens HERE, at process startup — never at import time. This is
    the concrete fix for the composition-root/import-time-loading flaw:
    importing src.main (or this module) must never trigger a real model
    load; only running the app through this lifespan does."""
    settings = load_settings()
    configure_logging(level=settings.log_level)

    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)
    retriever = HybridRetriever(vector_store, lexical_index)
    llm_client = GroqOpenAiLlmClient()

    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.query_use_case = QueryUseCase(retriever, llm_client, settings)
    app.state.rate_limiter = RateLimiter(max_requests=settings.rate_limit_per_minute)

    yield


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception",
        exc_info=exc,
        extra={"event": "unhandled_exception", "path": request.url.path},
    )
    return JSONResponse(status_code=500, content={"status": "error", "detail": "internal server error"})


def create_app() -> FastAPI:
    # A cheap env-var read (not a model load) — needed synchronously here for
    # CORS middleware, which must be added before lifespan runs.
    settings = load_settings()

    app = FastAPI(title="Manufacturing Knowledge RAG Assistant", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(router)
    configure_tracing(app)
    return app
