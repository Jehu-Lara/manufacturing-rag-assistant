"""Fail-first tests for Phase 2's `contextual-v1` index profile and the
candidate->swap rebuild safety (Task 5). These MUST fail until Task 6 adds
`ChromaVectorStore(..., index_profile=...)`, the heading-prefixed contextual
embedding input, and the candidate-swap rebuild. They use a recording fake
embedder — no model download.
"""

from __future__ import annotations

import chromadb
import pytest

from src.adapters.secondary.vector.chroma_vector_store import COLLECTION_NAME, ChromaVectorStore
from src.domain.models import ChunkMetadata
from tests.fakes import RecordingEmbedder


def _chunk(chunk_id: str, title: str, heading: str, text: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_title=title,
        revision="Rev A",
        section_heading=heading,
        source_type="synthetic",
        source_url_or_note="note",
        source_page_range=None,
        md_line_range="1-2",
        chunk_token_count=10,
        chunk_text=text,
    )


FIXTURE_CHUNKS = [
    _chunk(
        "alpha::0001",
        "Pump Maintenance Manual",
        "Priming Procedure",
        "Fill the casing with liquid before starting the pump.",
    ),
    _chunk(
        "beta::0001",
        "Hazard Communication SDS",
        "Handling and Storage",
        "Store containers in a cool, dry area away from oxidizers.",
    ),
]


def _expected_contextual(chunk: ChunkMetadata) -> str:
    """The single place the delimiter shape is written in this test module
    (controller ruling 1: ASCII `>`, exactly `title > heading\\n\\nbody`)."""
    return f"{chunk.document_title} > {chunk.section_heading}\n\n{chunk.chunk_text}"


def _store(tmp_path, embedder: RecordingEmbedder, profile: str | None = None) -> ChromaVectorStore:
    kwargs = {} if profile is None else {"index_profile": profile}
    return ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        embedder=embedder,
        collection_name=COLLECTION_NAME,
        **kwargs,
    )


def _client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


def _collection_names(client) -> set[str]:
    return {collection.name for collection in client.list_collections()}


def test_contextual_v1_embeds_heading_prefixed_strings_in_id_order(tmp_path):
    embedder = RecordingEmbedder()

    _store(tmp_path, embedder, profile="contextual-v1").build_collection(FIXTURE_CHUNKS)

    assert embedder.embed_texts_calls == [[_expected_contextual(c) for c in FIXTURE_CHUNKS]]


def test_raw_v1_embeds_unprefixed_chunk_text_bodies(tmp_path):
    embedder = RecordingEmbedder()

    _store(tmp_path, embedder, profile="raw-v1").build_collection(FIXTURE_CHUNKS)

    assert embedder.embed_texts_calls == [[c.chunk_text for c in FIXTURE_CHUNKS]]
    for recorded in embedder.embed_texts_calls[0]:
        assert " > " not in recorded


def test_default_profile_is_raw_v1(tmp_path):
    embedder = RecordingEmbedder()

    _store(tmp_path, embedder).build_collection(FIXTURE_CHUNKS)

    assert embedder.embed_texts_calls == [[c.chunk_text for c in FIXTURE_CHUNKS]]
    collection = _client(tmp_path).get_collection(COLLECTION_NAME)
    assert collection.metadata.get("index_profile") == "raw-v1"


@pytest.mark.parametrize("profile", ["raw-v1", "contextual-v1"])
def test_stored_document_and_metadata_are_byte_identical_raw_text(tmp_path, profile):
    embedder = RecordingEmbedder()

    _store(tmp_path, embedder, profile=profile).build_collection(FIXTURE_CHUNKS)

    collection = _client(tmp_path).get_collection(COLLECTION_NAME)
    stored = collection.get(
        ids=[c.chunk_id for c in FIXTURE_CHUNKS], include=["documents", "metadatas"]
    )
    by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
    for chunk in FIXTURE_CHUNKS:
        document, metadata = by_id[chunk.chunk_id]
        assert document == chunk.chunk_text
        assert metadata["chunk_text"] == chunk.chunk_text
        assert " > " not in document
        assert " > " not in metadata["chunk_text"]


def test_assert_fits_max_seq_length_receives_contextual_inputs_for_contextual_v1(tmp_path):
    embedder = RecordingEmbedder()

    _store(tmp_path, embedder, profile="contextual-v1").build_collection(FIXTURE_CHUNKS)

    assert embedder.assert_fits_calls == [[_expected_contextual(c) for c in FIXTURE_CHUNKS]]


def test_assert_fits_max_seq_length_receives_raw_bodies_for_raw_v1(tmp_path):
    embedder = RecordingEmbedder()

    _store(tmp_path, embedder, profile="raw-v1").build_collection(FIXTURE_CHUNKS)

    assert embedder.assert_fits_calls == [[c.chunk_text for c in FIXTURE_CHUNKS]]


def test_failed_candidate_build_leaves_live_collection_queryable_and_unchanged(tmp_path):
    _store(tmp_path, RecordingEmbedder()).build_collection(FIXTURE_CHUNKS)

    with pytest.raises(RuntimeError):
        _store(tmp_path, RecordingEmbedder(fail_on_embed=True)).build_collection(FIXTURE_CHUNKS)

    live = _client(tmp_path).get_collection(COLLECTION_NAME)
    stored = live.get(ids=[c.chunk_id for c in FIXTURE_CHUNKS], include=["documents"])
    assert set(stored["ids"]) == {c.chunk_id for c in FIXTURE_CHUNKS}
    assert live.count() == len(FIXTURE_CHUNKS)


def test_stale_candidate_collection_is_discarded_before_new_build(tmp_path):
    client = _client(tmp_path)
    stale = client.create_collection(
        f"{COLLECTION_NAME}__candidate", metadata={"hnsw:space": "cosine"}
    )
    stale.add(ids=["stale::9999"], embeddings=[[0.0, 0.0]], documents=["stale candidate row"])

    _store(tmp_path, RecordingEmbedder()).build_collection(FIXTURE_CHUNKS)

    names = _collection_names(_client(tmp_path))
    assert f"{COLLECTION_NAME}__candidate" not in names
    assert f"{COLLECTION_NAME}__previous" not in names
    live = _client(tmp_path).get_collection(COLLECTION_NAME)
    assert set(live.get()["ids"]) == {c.chunk_id for c in FIXTURE_CHUNKS}


def test_successful_rebuild_swaps_cleanly_leaving_no_side_collections(tmp_path):
    store = _store(tmp_path, RecordingEmbedder())
    store.build_collection(FIXTURE_CHUNKS[:1])
    store.build_collection(FIXTURE_CHUNKS)

    client = _client(tmp_path)
    assert _collection_names(client) == {COLLECTION_NAME}
    live = client.get_collection(COLLECTION_NAME)
    assert live.count() == len(FIXTURE_CHUNKS)
    assert live.metadata.get("index_profile") == "raw-v1"
