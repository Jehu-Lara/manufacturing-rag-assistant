from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from src.domain.models import ChunkMetadata, RetrievalResult


@runtime_checkable
class EmbedderPort(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def max_seq_length(self) -> int: ...
    def assert_fits_max_seq_length(self, texts: list[str]) -> None: ...


@runtime_checkable
class VectorStorePort(Protocol):
    def build_collection(self, chunks: list[ChunkMetadata], embedding_inputs: list[str]) -> None: ...
    def query(self, text: str, top_n: int) -> list[tuple[str, float, dict[str, Any]]]: ...
    def get_metadata(self, chunk_id: str) -> dict[str, Any]: ...
    def ping(self) -> bool: ...


@runtime_checkable
class LexicalIndexPort(Protocol):
    def build_index(self, chunks: list[ChunkMetadata], *, chunks_sha256: str) -> None: ...
    def query(self, text: str, top_n: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class RerankerPort(Protocol):
    """`candidates` is (chunk_id, text); the return is (chunk_id, score)
    best-first over EXACTLY the same id set. Adding, dropping or deduplicating
    an id is a contract violation the retriever rejects, because the refusal
    gate's guarantee rests on the semantic_rank == 1 result still being
    present."""

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]: ...


@runtime_checkable
class RetrieverPort(Protocol):
    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]: ...


@runtime_checkable
class LLMClientPort(Protocol):
    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...
