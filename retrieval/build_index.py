from __future__ import annotations

import json
from pathlib import Path

from ingestion.metadata import ChunkMetadata
from retrieval import bm25_index, embedder, vector_store

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "ingestion" / "output" / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS_FILE) -> list[ChunkMetadata]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m ingestion.run` first to produce it")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(ChunkMetadata(**json.loads(line)))
    return chunks


def run() -> None:
    chunks = load_chunks()

    embedder.assert_fits_max_seq_length([chunk.chunk_text for chunk in chunks])

    vector_store.build_collection(chunks)
    bm25_index.build_index(chunks)

    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding model: {embedder.MODEL_NAME} (max_seq_length={embedder.max_seq_length()})")
    print(f"Vector store: {vector_store.PERSIST_DIR} (collection '{vector_store.COLLECTION_NAME}')")
    print(f"BM25 index: {bm25_index.PERSIST_FILE}")


if __name__ == "__main__":
    run()
