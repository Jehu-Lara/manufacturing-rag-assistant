from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.features.retrieval.cli as cli_module
from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.vector.chroma_vector_store import contextual_embedding_input
from src.domain.models import ChunkMetadata
from src.features.retrieval import index_manifest
from src.features.retrieval.cli import load_chunks
from tests.fakes import RecordingEmbedder


@pytest.mark.parametrize("profile", ["raw-v1", "contextual-v1"])
def test_model_max_seq_length_covers_every_real_corpus_chunk(profile: str):
    # Guards a model disqualified by its OWN tokenizer's max_seq_length,
    # checked against real chunks — not against tiktoken's cl100k_base count
    # (which disagrees with this model's tokenizer and is not what would
    # actually truncate). contextual-v1 (the shipped default) embeds the longer
    # heading-prefixed input, so both profiles' inputs must fit.
    chunks = load_chunks()
    embedder = SentenceTransformersEmbedder()
    inputs = (
        [contextual_embedding_input(c) for c in chunks]
        if profile == "contextual-v1"
        else [c.chunk_text for c in chunks]
    )
    embedder.assert_fits_max_seq_length(inputs)


def test_tokenized_length_is_positive_for_nonempty_text():
    embedder = SentenceTransformersEmbedder()
    assert embedder.tokenized_length("hello world") > 0


def test_max_seq_length_is_a_positive_integer():
    embedder = SentenceTransformersEmbedder()
    assert isinstance(embedder.max_seq_length(), int)
    assert embedder.max_seq_length() > 0


# --- Fail-first: CLI index-profile selection + manifest emission (Task 6 makes these pass) ---


def _cli_chunk(chunk_id: str, title: str, heading: str, text: str) -> ChunkMetadata:
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


_CLI_FIXTURE_CHUNKS = [
    _cli_chunk("alpha::0001", "Pump Maintenance Manual", "Priming Procedure", "Fill the casing first."),
    _cli_chunk("beta::0001", "Hazard Communication SDS", "Handling and Storage", "Keep away from oxidizers."),
]


class _SpyVectorStore:
    instances: list[_SpyVectorStore] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.built_chunks: list[ChunkMetadata] | None = None
        _SpyVectorStore.instances.append(self)

    def build_collection(self, chunks: list[ChunkMetadata]) -> None:
        self.built_chunks = list(chunks)


class _SpyLexical:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def build_index(self, chunks: list[ChunkMetadata]) -> None:
        pass


class _StubManifest:
    def __init__(self, index_profile: str, chunk_count: int) -> None:
        self.index_profile = index_profile
        self.chunk_count = chunk_count


def _patch_cli(monkeypatch, tmp_path):
    _SpyVectorStore.instances = []
    written: list[_StubManifest] = []
    embedder = RecordingEmbedder()
    monkeypatch.setattr(
        cli_module,
        "load_settings",
        lambda: SimpleNamespace(chroma_path=tmp_path / "chroma", bm25_path=tmp_path / "bm25_index.json"),
    )
    monkeypatch.setattr(cli_module, "load_chunks", lambda: list(_CLI_FIXTURE_CHUNKS))
    monkeypatch.setattr(cli_module, "SentenceTransformersEmbedder", lambda: embedder)
    monkeypatch.setattr(cli_module, "ChromaVectorStore", _SpyVectorStore)
    monkeypatch.setattr(cli_module, "Bm25LexicalIndex", _SpyLexical)
    monkeypatch.setattr(
        index_manifest,
        "build_manifest",
        lambda index_profile, chunk_count, **kw: _StubManifest(index_profile, chunk_count),
    )
    monkeypatch.setattr(index_manifest, "write", lambda manifest, *a, **k: written.append(manifest))
    return written


def test_run_defaults_index_profile_to_contextual_v1(monkeypatch, tmp_path):
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.delenv("INDEX_PROFILE", raising=False)

    cli_module.run()

    assert _SpyVectorStore.instances[-1].kwargs.get("index_profile") == "contextual-v1"


def test_run_reads_index_profile_env_for_raw_v1_rollback(monkeypatch, tmp_path):
    written = _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("INDEX_PROFILE", "raw-v1")

    cli_module.run()

    assert _SpyVectorStore.instances[-1].kwargs.get("index_profile") == "raw-v1"
    assert written[-1].index_profile == "raw-v1"


def test_run_reads_index_profile_env_for_contextual_v1(monkeypatch, tmp_path):
    written = _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("INDEX_PROFILE", "contextual-v1")

    cli_module.run()

    assert _SpyVectorStore.instances[-1].kwargs.get("index_profile") == "contextual-v1"
    assert written[-1].index_profile == "contextual-v1"


def test_run_rejects_invalid_index_profile(monkeypatch, tmp_path):
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("INDEX_PROFILE", "totally-bogus")

    with pytest.raises(ValueError):
        cli_module.run()


def test_run_emits_index_manifest_via_build_and_write(monkeypatch, tmp_path):
    written = _patch_cli(monkeypatch, tmp_path)
    monkeypatch.delenv("INDEX_PROFILE", raising=False)

    cli_module.run()

    assert len(written) == 1
    assert written[0].index_profile == "contextual-v1"
    assert written[0].chunk_count == len(_CLI_FIXTURE_CHUNKS)
