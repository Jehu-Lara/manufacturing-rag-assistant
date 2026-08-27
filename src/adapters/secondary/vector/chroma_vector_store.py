from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb

from src.domain.models import ChunkMetadata
from src.domain.ports import EmbedderPort

COLLECTION_NAME = "manufacturing_chunks"


class ChromaVectorStore:
    """Implements VectorStorePort."""

    def __init__(self, persist_dir: Path, embedder: EmbedderPort, collection_name: str = COLLECTION_NAME) -> None:
        self._persist_dir = persist_dir
        self._embedder = embedder
        self._collection_name = collection_name

    def _client(self) -> Any:
        # chromadb ships a py.typed marker but mypy can't statically resolve
        # chromadb.ClientAPI as an annotation here; Any is the honest type
        # for this internal helper rather than fighting the stub.
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._persist_dir))

    def _to_chroma_metadata(self, chunk: ChunkMetadata) -> dict[str, Any]:
        metadata = chunk.to_dict()
        # Chroma metadata values must be str/int/float/bool — None isn't
        # accepted. source_page_range is the one field where None is a valid,
        # expected value (every synthetic-doc chunk). Omitting the key
        # (rather than coercing to "") preserves the same information:
        # callers read it back with .get(...), which returns None either way.
        if metadata["source_page_range"] is None:
            del metadata["source_page_range"]
        return metadata

    def build_collection(self, chunks: list[ChunkMetadata]) -> None:
        client = self._client()
        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass
        collection = client.create_collection(self._collection_name, metadata={"hnsw:space": "cosine"})

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.chunk_text for chunk in chunks]
        embeddings = self._embedder.embed_texts(documents)
        metadatas = [self._to_chroma_metadata(chunk) for chunk in chunks]

        # chromadb's stub wants numpy-array embeddings and a narrower metadata
        # value union than plain list[list[float]]/list[dict] give it; both
        # are runtime-valid inputs chromadb accepts directly, just not what
        # the stub states.
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def _get_collection(self) -> Any:
        return self._client().get_collection(self._collection_name)

    def query(self, text: str, top_n: int) -> list[tuple[str, float, dict[str, Any]]]:
        """Returns (chunk_id, cosine_similarity, metadata) tuples, best match first."""
        collection = self._get_collection()
        result = collection.query(query_embeddings=[self._embedder.embed_query(text)], n_results=top_n)
        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        return [(chunk_id, 1.0 - distance, metadata) for chunk_id, distance, metadata in zip(ids, distances, metadatas)]

    def get_metadata(self, chunk_id: str) -> dict[str, Any]:
        collection = self._get_collection()
        result = collection.get(ids=[chunk_id])
        if not result["ids"]:
            raise KeyError(f"chunk_id {chunk_id!r} not found in vector store")
        return cast("dict[str, Any]", result["metadatas"][0])

    def ping(self) -> bool:
        try:
            self._get_collection()
            return True
        except Exception:
            return False
