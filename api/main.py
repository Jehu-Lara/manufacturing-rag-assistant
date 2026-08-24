from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

import api.generation
import api.logging_setup
import retrieval.vector_store
from api.config import Settings, load_settings
from api.schemas import HealthResponse, QueryRequest, QueryResponse
from retrieval.embedder import MODEL_NAME

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    api.logging_setup.configure(level=settings.log_level)
    app.state.settings = settings
    yield


app = FastAPI(title="Manufacturing Knowledge RAG Assistant", lifespan=lifespan)


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
def query(request: QueryRequest) -> QueryResponse:
    settings: Settings = app.state.settings
    return api.generation.answer_question(request.question, request.language, settings)
