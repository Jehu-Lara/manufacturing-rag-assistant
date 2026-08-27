from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MAX_QUESTION_LENGTH = 2000


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    language: Literal["en", "es"]


class Citation(BaseModel):
    document_id: str
    document_title: str
    section_heading: str
    revision: str
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    status: Literal["ok", "error"]
    confidence: Optional[float]
    threshold: float
    language: Literal["en", "es"]
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    embedding_model: str
    llm_provider_primary: str
    index_loaded: bool


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
