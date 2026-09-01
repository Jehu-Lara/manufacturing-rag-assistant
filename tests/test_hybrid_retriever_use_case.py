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


def test_known_query_returns_known_relevant_chunk_in_top_k():
    # Integration test against the real built index — requires
    # `python -m src.features.retrieval.cli` to have been run first.
    from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
    from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
    from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
    from src.core.config import load_settings

    settings = load_settings()
    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)
    retriever = HybridRetriever(vector_store, lexical_index)

    results = retriever.retrieve("What is lockout/tagout and why does it matter?", k=3)
    retrieved_ids = [r.chunk_id for r in results]
    assert any(chunk_id.startswith("osha-3120-lockout-tagout::") for chunk_id in retrieved_ids), (
        f"expected a lockout/tagout chunk in top-3, got {retrieved_ids}"
    )


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


class _SpyVectorStore(_StubVectorStore):
    def query(self, text: str, top_n: int) -> list[tuple[str, float, dict[str, Any]]]:
        self.last_query = text
        return super().query(text, top_n)


class _SpyLexicalIndex(_StubLexicalIndex):
    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        self.last_query = text
        return super().query(text, top_n)


def _spies() -> tuple[_SpyVectorStore, _SpyLexicalIndex]:
    md = {"c": {"document_id": "d"}}
    vs = _SpyVectorStore(hits=[("c", 0.9, md["c"])], metadata_by_id=md)
    lx = _SpyLexicalIndex(hits=[("c", 1.0)])
    return vs, lx


def test_expansion_mode_off_passes_original_to_both():
    vs, lx = _spies()
    HybridRetriever(vs, lx).retrieve("What is NPSHA?", k=1)
    assert vs.last_query == "What is NPSHA?"
    assert lx.last_query == "What is NPSHA?"


def test_expansion_mode_semantic_expands_vector_only():
    vs, lx = _spies()
    HybridRetriever(vs, lx, expansion_mode="semantic").retrieve("What is NPSHA?", k=1)
    assert "net positive suction head available" in vs.last_query
    assert lx.last_query == "What is NPSHA?"


def test_expansion_mode_lexical_expands_bm25_only():
    vs, lx = _spies()
    HybridRetriever(vs, lx, expansion_mode="lexical").retrieve("What is NPSHA?", k=1)
    assert vs.last_query == "What is NPSHA?"
    assert "net positive suction head available" in lx.last_query


def test_expansion_mode_both_expands_both():
    vs, lx = _spies()
    HybridRetriever(vs, lx, expansion_mode="both").retrieve("What is NPSHA?", k=1)
    assert "net positive suction head available" in vs.last_query
    assert "net positive suction head available" in lx.last_query
