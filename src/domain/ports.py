from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.core.config import Settings
from src.domain.models import ChunkMetadata, RetrievalResult


@runtime_checkable
class EmbedderPort(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def max_seq_length(self) -> int: ...
    def assert_fits_max_seq_length(self, texts: list[str]) -> None: ...


@runtime_checkable
class VectorStorePort(Protocol):
    def build_collection(self, chunks: list[ChunkMetadata]) -> None: ...
    def query(self, text: str, top_n: int) -> list[tuple[str, float, dict[str, Any]]]: ...
    def get_metadata(self, chunk_id: str) -> dict[str, Any]: ...
    def ping(self) -> bool: ...


@runtime_checkable
class LexicalIndexPort(Protocol):
    def build_index(self, chunks: list[ChunkMetadata]) -> None: ...
    def query(self, text: str, top_n: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class RetrieverPort(Protocol):
    def retrieve(self, query_text: str, k: int, top_n: int) -> list[RetrievalResult]: ...


@runtime_checkable
class LLMClientPort(Protocol):
    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], settings: Settings
    ) -> dict[str, Any]: ...
