"""CI-safe end-to-end smoke test for the ingestion+index-build CLI chain,
run against tests/fixtures/mini_corpus/ instead of the real ~2GB corpus
(Phase 3 gate: "ingestion+index-build CLIs runnable against a small test
corpus, not the full corpus in CI")."""

from __future__ import annotations

from pathlib import Path

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.domain.policies import embedding_inputs
from src.features.ingestion.chunker import TiktokenCounter
from src.features.ingestion.cli import build_chunks_for_document
from src.features.ingestion.use_cases import load_corpus
from src.features.retrieval.use_cases import HybridRetriever

MINI_CORPUS_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_corpus"


def test_ingestion_and_index_build_against_mini_corpus(tmp_path):
    documents = load_corpus(corpus_root=MINI_CORPUS_ROOT)
    assert len(documents) == 2

    counter = TiktokenCounter()
    chunks = []
    for document in documents:
        chunks.extend(build_chunks_for_document(document, counter))
    assert chunks, "expected at least one chunk from the mini corpus"

    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma", embedder=embedder, collection_name="mini_test")
    lexical_index = Bm25LexicalIndex(persist_path=tmp_path / "bm25_index.json")

    vector_store.build_collection(chunks, embedding_inputs(chunks, vector_store._index_profile))
    # The mini corpus is chunked in memory, with no chunks.jsonl on disk, so
    # there is nothing to hash here — any stable digest satisfies the contract
    # that the caller states which chunk set this index was built from.
    lexical_index.build_index(chunks, chunks_sha256="mini-corpus-smoke")

    retriever = HybridRetriever(vector_store, lexical_index)
    results = retriever.retrieve("smoke-test the ingestion and retrieval CLIs", k=3)

    assert results
    assert any(
        r.chunk_id.startswith("mini-public-doc::") or r.chunk_id.startswith("mini-synthetic-doc::") for r in results
    )
