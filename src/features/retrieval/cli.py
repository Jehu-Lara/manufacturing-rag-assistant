from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
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

    vector_store.build_collection(chunks)
    lexical_index.build_index(chunks)

    manifest = index_manifest.build_manifest(profile, len(chunks))
    index_manifest.write(manifest)

    print(f"Index profile: {profile}")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME} (max_seq_length={embedder.max_seq_length()})")
    print(f"Vector store: {settings.chroma_path}")
    print(f"BM25 index: {settings.bm25_path}")
    print(f"Index manifest: {index_manifest.MANIFEST_FILE}")


if __name__ == "__main__":
    run()
