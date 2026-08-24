from __future__ import annotations

from pathlib import Path

import chromadb

from ingestion.metadata import ChunkMetadata
from retrieval.embedder import embed_query, embed_texts

PERSIST_DIR = Path(__file__).resolve().parent / "output" / "chroma"
COLLECTION_NAME = "manufacturing_chunks"


def _client() -> chromadb.ClientAPI:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


def _to_chroma_metadata(chunk: ChunkMetadata) -> dict:
    metadata = chunk.to_dict()
    # Chroma metadata values must be str/int/float/bool — None isn't accepted.
    # source_page_range is the one field where None is a valid, expected value
    # (every synthetic-doc chunk). Omitting the key (rather than coercing to
    # "") preserves the same information: callers read it back with .get(...),
    # which returns None either way.
    if metadata["source_page_range"] is None:
        del metadata["source_page_range"]
    return metadata


def build_collection(chunks: list[ChunkMetadata]) -> None:
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.chunk_text for chunk in chunks]
    embeddings = embed_texts(documents)
    metadatas = [_to_chroma_metadata(chunk) for chunk in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def get_collection():
    return _client().get_collection(COLLECTION_NAME)


def query(text: str, top_n: int) -> list[tuple[str, float, dict]]:
    """Returns (chunk_id, cosine_similarity, metadata) tuples, best match first."""
    collection = get_collection()
    result = collection.query(query_embeddings=[embed_query(text)], n_results=top_n)
    ids = result["ids"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]
    return [(chunk_id, 1.0 - distance, metadata) for chunk_id, distance, metadata in zip(ids, distances, metadatas)]


def get_metadata(chunk_id: str) -> dict:
    collection = get_collection()
    result = collection.get(ids=[chunk_id])
    if not result["ids"]:
        raise KeyError(f"chunk_id {chunk_id!r} not found in vector store")
    return result["metadatas"][0]
