from __future__ import annotations

import json
from pathlib import Path

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.models import ChunkMetadata

CHUNKS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "ingestion" / "output" / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS_FILE) -> list[ChunkMetadata]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.features.ingestion.cli` first to produce it")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(ChunkMetadata(**json.loads(line)))
    return chunks


def run() -> None:
    settings = load_settings()
    chunks = load_chunks()

    embedder = SentenceTransformersEmbedder()
    embedder.assert_fits_max_seq_length([chunk.chunk_text for chunk in chunks])

    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)

    vector_store.build_collection(chunks)
    lexical_index.build_index(chunks)

    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME} (max_seq_length={embedder.max_seq_length()})")
    print(f"Vector store: {settings.chroma_path}")
    print(f"BM25 index: {settings.bm25_path}")


if __name__ == "__main__":
    run()
