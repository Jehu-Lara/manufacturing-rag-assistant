from __future__ import annotations

from typing import Any, Callable, Optional, Union

from src.domain.models import RetrievalResult


class InMemoryRetriever:
    """Implements RetrieverPort with a fixed, pre-programmed result list —
    no ChromaDB, no BM25, no embedding model."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        return self._results[:k]


class InMemoryLLMClient:
    """Implements LLMClientPort with a pre-programmed response (dict or a
    callable producing one) or a raised exception — no real provider SDK
    call, no network."""

    def __init__(
        self,
        response: Optional[Union[dict[str, Any], Callable[[], dict[str, Any]]]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._error = error

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], settings: Any
    ) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        if callable(self._response):
            return self._response()
        assert self._response is not None
        return self._response


class InMemoryVectorStore:
    """Implements just enough of VectorStorePort for /health and /ready
    tests: a fixed ping() outcome, no real ChromaDB connection."""

    def __init__(self, ready: bool = True) -> None:
        self._ready = ready

    def build_collection(self, chunks: list[Any]) -> None:
        raise NotImplementedError

    def query(self, text: str, top_n: int) -> list[tuple[str, float, dict[str, Any]]]:
        raise NotImplementedError

    def get_metadata(self, chunk_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def ping(self) -> bool:
        return self._ready
