from __future__ import annotations

import json
import os
from pathlib import Path

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.models import ChunkMetadata, IndexProfile
from src.features.retrieval import index_manifest

CHUNKS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "ingestion" / "output" / "chunks.jsonl"

_VALID_PROFILES: tuple[IndexProfile, ...] = ("raw-v1", "contextual-v1")


def load_chunks(path: Path = CHUNKS_FILE) -> list[ChunkMetadata]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.features.ingestion.cli` first to produce it")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(ChunkMetadata(**json.loads(line)))
    return chunks


def run() -> None:
    profile = os.environ.get("INDEX_PROFILE", "raw-v1")
    if profile not in _VALID_PROFILES:
        raise ValueError(f"INDEX_PROFILE must be one of {_VALID_PROFILES}, got {profile!r}")

    settings = load_settings()
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
