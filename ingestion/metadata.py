from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

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

    def to_dict(self) -> dict:
        return asdict(self)
