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


class RecordingEmbedder:
    """Implements EmbedderPort, recording the exact `texts` lists passed to
    embed_texts / assert_fits_max_seq_length so tests can assert on them.
    Deterministic vectors, no sentence-transformers model load."""

    def __init__(self, *, fail_on_embed: bool = False, max_seq: int = 8192) -> None:
        self.embed_texts_calls: list[list[str]] = []
        self.assert_fits_calls: list[list[str]] = []
        self._fail_on_embed = fail_on_embed
        self._max_seq = max_seq

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        recorded = list(texts)
        self.embed_texts_calls.append(recorded)
        if self._fail_on_embed:
            raise RuntimeError("simulated embedding failure")
        return [[float(len(text)), float(i)] for i, text in enumerate(recorded)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]

    def max_seq_length(self) -> int:
        return self._max_seq

    def assert_fits_max_seq_length(self, texts: list[str]) -> None:
        self.assert_fits_calls.append(list(texts))


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
