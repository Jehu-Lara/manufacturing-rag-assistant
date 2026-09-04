from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import LEXICAL_PROFILE, Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.policies import embedding_inputs
from src.features.retrieval import index_manifest
from src.features.retrieval.chunk_store import CHUNKS_FILE, load_chunks

__all__ = ["CHUNKS_FILE", "load_chunks", "run"]


def run() -> None:
    settings = load_settings()
    profile = index_manifest.resolve_index_profile(settings)
    chunks = load_chunks()

    embedder = SentenceTransformersEmbedder()

    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder, index_profile=profile)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)

    # One digest for both artifacts, so the BM25 payload and the manifest can
    # never disagree about which chunks.jsonl this index was built from.
    digest = index_manifest.chunks_sha256()

    vector_store.build_collection(chunks, embedding_inputs(chunks, profile))
    lexical_index.build_index(chunks, chunks_sha256=digest)

    # The manifest is written LAST, as the commit marker: its presence asserts
    # that both artifacts were built and promoted. Writing it earlier would let
    # a BM25 failure leave a manifest describing an index that does not exist.
    # On failure the previous manifest stays, so verify() fails loudly at the
    # next startup instead of half-passing.
    index_manifest.write(index_manifest.build_manifest(profile, len(chunks)))

    print(f"Index profile: {profile}")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME} (max_seq_length={embedder.max_seq_length()})")
    print(f"Vector store: {settings.chroma_path}")
    print(f"BM25 index: {settings.bm25_path} (lexical_profile={LEXICAL_PROFILE})")
    print(f"Index manifest: {index_manifest.MANIFEST_FILE}")


if __name__ == "__main__":
    run()
