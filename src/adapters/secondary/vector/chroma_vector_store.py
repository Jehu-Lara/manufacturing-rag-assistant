from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb

from src.domain.models import ChunkMetadata, IndexProfile
from src.domain.policies import embedding_input
from src.domain.ports import EmbedderPort

COLLECTION_NAME = "manufacturing_chunks"


def contextual_embedding_input(chunk: ChunkMetadata) -> str:
    """Deprecated shim — the policy moved to src.domain.policies.embedding_input
    so a second vector store could not silently disagree with this one about
    what contextual-v1 means. Kept for one release because tests and external
    readers import it from here."""
    return embedding_input(chunk, "contextual-v1")


class ChromaVectorStore:
    """Implements VectorStorePort."""

    def __init__(
        self,
        persist_dir: Path,
        embedder: EmbedderPort,
        collection_name: str = COLLECTION_NAME,
        *,
        index_profile: IndexProfile = "raw-v1",
    ) -> None:
        self._persist_dir = persist_dir
        self._embedder = embedder
        self._collection_name = collection_name
        self._index_profile: IndexProfile = index_profile

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

    def build_collection(self, chunks: list[ChunkMetadata], embedding_inputs: list[str]) -> None:
        """A writer, not a policy holder: it embeds the strings the caller
        computed (src.domain.policies.embedding_inputs) and stores the raw
        chunk_text regardless of profile (ADR-008)."""
        if len(embedding_inputs) != len(chunks):
            raise ValueError(
                f"embedding_inputs has {len(embedding_inputs)} entries, expected {len(chunks)}"
            )
        client = self._client()

        # Validate + embed BEFORE any collection is created/deleted/renamed, so
        # a length or model failure leaves the live collection untouched.
        self._embedder.assert_fits_max_seq_length(embedding_inputs)
        embeddings = self._embedder.embed_texts(embedding_inputs)

        candidate_name = f"{self._collection_name}__candidate"
        previous_name = f"{self._collection_name}__previous"

        try:
            client.delete_collection(candidate_name)
        except chromadb.errors.NotFoundError:
            pass

        candidate = client.create_collection(
            candidate_name,
            metadata={"hnsw:space": "cosine", "index_profile": self._index_profile},
        )
        # documents and metadatas are the RAW chunk text for both profiles — only
        # the embedding vectors differ. chromadb's stub wants numpy-array
        # embeddings and a narrower metadata value union than plain
        # list[list[float]]/list[dict]; both are runtime-valid inputs.
        candidate.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.chunk_text for chunk in chunks],
            metadatas=[self._to_chroma_metadata(chunk) for chunk in chunks],
        )
        if candidate.count() != len(chunks):
            raise RuntimeError(
                f"candidate collection has {candidate.count()} rows, expected {len(chunks)}"
            )

        self._promote(client, candidate, previous_name)

    def _promote(self, client: Any, candidate: Any, previous_name: str) -> None:
        try:
            client.delete_collection(previous_name)
        except chromadb.errors.NotFoundError:
            pass

        try:
            live = client.get_collection(self._collection_name)
        except chromadb.errors.NotFoundError:
            live = None

        if live is not None:
            live.modify(name=previous_name)

        try:
            candidate.modify(name=self._collection_name)
        except Exception:
            if live is not None:
                client.get_collection(previous_name).modify(name=self._collection_name)
            raise

        client.get_collection(self._collection_name)

        if live is not None:
            try:
                client.delete_collection(previous_name)
            except chromadb.errors.NotFoundError:
                pass

    def _get_collection(self) -> Any:
        return self._client().get_collection(self._collection_name)

    def validate_collection(self, *, expected_profile: IndexProfile, expected_count: int) -> None:
        """Startup guard: the live collection must carry the expected
        index_profile and row count, or the app is about to serve a stale or
        wrong-profile index."""
        collection = self._get_collection()
        metadata = collection.metadata or {}
        actual_profile = metadata.get("index_profile")
        if actual_profile != expected_profile:
            raise RuntimeError(
                f"live collection index_profile is {actual_profile!r}, expected {expected_profile!r}"
            )
        actual_count = collection.count()
        if actual_count != expected_count:
            raise RuntimeError(
                f"live collection has {actual_count} rows, expected {expected_count}"
            )

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
