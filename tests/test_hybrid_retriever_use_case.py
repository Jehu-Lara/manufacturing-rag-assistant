from __future__ import annotations

from typing import Any

from src.features.retrieval.use_cases import DEFAULT_TOP_N, SEMANTIC_EXTRACTION_K, HybridRetriever


class _StubVectorStore:
    def __init__(
        self, hits: list[tuple[str, float, dict[str, Any]]], metadata_by_id: dict[str, dict[str, Any]]
    ) -> None:
        self._hits = hits
        self._metadata_by_id = metadata_by_id

    def build_collection(self, chunks: list[Any]) -> None:
        raise NotImplementedError

    def query(self, text: str, top_n: int) -> list[tuple[str, float, dict[str, Any]]]:
        return self._hits[:top_n]

    def get_metadata(self, chunk_id: str) -> dict[str, Any]:
        return self._metadata_by_id[chunk_id]

    def ping(self) -> bool:
        return True


class _StubLexicalIndex:
    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits

    def build_index(self, chunks: list[Any]) -> None:
        raise NotImplementedError

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        return self._hits[:top_n]


def test_semantic_extraction_k_is_double_default_top_n():
    assert SEMANTIC_EXTRACTION_K == DEFAULT_TOP_N * 2


def test_hybrid_retriever_favors_item_ranked_highly_in_both_lists():
    metadata = {
        "shared": {"document_id": "doc-shared"},
        "semantic_only": {"document_id": "doc-semantic"},
        "other": {"document_id": "doc-other"},
    }
    vector_store = _StubVectorStore(
        hits=[("semantic_only", 0.9, metadata["semantic_only"]), ("shared", 0.8, metadata["shared"])],
        metadata_by_id=metadata,
    )
    lexical_index = _StubLexicalIndex(hits=[("other", 5.0), ("shared", 4.0)])
    retriever = HybridRetriever(vector_store, lexical_index)

    results = retriever.retrieve("some query", k=3)

    assert results[0].chunk_id == "shared"
    assert results[0].semantic_rank == 2
    assert results[0].bm25_rank == 2


def test_hybrid_retriever_uses_vector_store_metadata_for_bm25_only_hits():
    metadata = {"bm25-only": {"document_id": "doc-bm25-only"}}
    vector_store = _StubVectorStore(hits=[], metadata_by_id=metadata)
    lexical_index = _StubLexicalIndex(hits=[("bm25-only", 3.0)])
    retriever = HybridRetriever(vector_store, lexical_index)

    results = retriever.retrieve("some query", k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "bm25-only"
    assert results[0].semantic_rank is None
    assert results[0].metadata == metadata["bm25-only"]


def test_hybrid_retriever_truncates_to_k():
    metadata = {f"chunk-{i}": {"document_id": f"doc-{i}"} for i in range(5)}
    vector_store = _StubVectorStore(
        hits=[(f"chunk-{i}", 1.0 - i * 0.1, metadata[f"chunk-{i}"]) for i in range(5)],
        metadata_by_id=metadata,
    )
    lexical_index = _StubLexicalIndex(hits=[])
    retriever = HybridRetriever(vector_store, lexical_index)

    results = retriever.retrieve("some query", k=2)

    assert len(results) == 2
