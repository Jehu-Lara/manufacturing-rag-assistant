from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_LENGTH = 2000


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    language: Literal["en", "es"]

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """min_length=1 passes "   ". Retrieval would then embed whitespace and
        the refusal gate would score noise as if it were a question."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must contain non-whitespace characters")
        return stripped


class Citation(BaseModel):
    document_id: str
    document_title: str
    section_heading: str
    revision: str
    chunk_id: str
    source_type: Literal["public", "synthetic"]


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    status: Literal["ok", "error"]
    confidence: Optional[float]
    threshold: float
    review_floor: Optional[float]
    gate_band: Literal["hard_refuse", "grounded_review", "confident"]
    language: Literal["en", "es"]
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    embedding_model: str
    llm_provider_primary: str
    index_loaded: bool


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
