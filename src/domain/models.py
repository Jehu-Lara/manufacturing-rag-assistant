from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

Language = Literal["en", "es"]
ExpansionMode = Literal["off", "semantic", "lexical", "both"]
IndexProfile = Literal["raw-v1", "contextual-v1"]
SourceType = Literal["public", "synthetic"]

_REQUIRED_STRING_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "revision",
    "section_heading",
    "source_type",
    "source_url_or_note",
    "md_line_range",
    "chunk_text",
)


@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: str
    document_id: str
    document_title: str
    revision: str
    section_heading: str
    source_type: SourceType
    source_url_or_note: str
    source_page_range: Optional[str]
    md_line_range: str
    chunk_token_count: int
    chunk_text: str

    def validate(self) -> None:
        for field_name in _REQUIRED_STRING_FIELDS:
            value = getattr(self, field_name)
            if not value or not str(value).strip():
                label = self.chunk_id or "<unknown chunk>"
                raise ValueError(f"{label}: missing required field '{field_name}'")
        if self.source_type not in ("public", "synthetic"):
            raise ValueError(f"{self.chunk_id}: invalid source_type {self.source_type!r}")
        if self.chunk_token_count <= 0:
            raise ValueError(f"{self.chunk_id}: non-positive chunk_token_count {self.chunk_token_count}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    fused_score: float
    semantic_rank: Optional[int]
    semantic_score: Optional[float]
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Citation:
    document_id: str
    document_title: str
    section_heading: str
    revision: str
    chunk_id: str


@dataclass(frozen=True)
class QueryAnswer:
    answer: str
    citations: list[Citation]
    refused: bool
    status: Literal["ok", "error"]
    confidence: Optional[float]
    threshold: float
    language: Language
    request_id: str
