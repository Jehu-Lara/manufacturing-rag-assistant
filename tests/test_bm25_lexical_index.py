from __future__ import annotations

import json

import pytest

from src.adapters.secondary.lexical.bm25_lexical_index import (
    BM25_SCHEMA_VERSION,
    LEXICAL_PROFILE,
    Bm25LexicalIndex,
)
from src.domain.models import ChunkMetadata


def _chunk(chunk_id: str, text: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_title=f"Title {chunk_id}",
        revision="Rev A",
        section_heading="Section",
        source_type="synthetic",
        source_url_or_note="note",
        source_page_range=None,
        md_line_range="1-2",
        chunk_token_count=10,
        chunk_text=text,
    )


def test_build_index_persists_plain_json_not_pickle(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    index = Bm25LexicalIndex(persist_path)

    index.build_index(
        [_chunk("chunk-1", "lockout tagout safety procedure"), _chunk("chunk-2", "quality control unit")],
        chunks_sha256="deadbeef",
    )

    raw = persist_path.read_text(encoding="utf-8")
    data = json.loads(raw)  # would raise if it were pickle bytes, not JSON
    assert data["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert data["corpus_tokens"][0] == ["lockout", "tagout", "safety", "procedure"]


def test_query_ranks_matching_document_first(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    index = Bm25LexicalIndex(persist_path)
    index.build_index(
        [
            _chunk("chunk-lockout", "lockout tagout safety procedure for machinery"),
            _chunk("chunk-quality", "quality control unit responsibilities"),
        ],
        chunks_sha256="test",
    )

    results = index.query("lockout tagout", top_n=2)

    assert results[0][0] == "chunk-lockout"


def test_query_loads_from_a_fresh_instance_pointed_at_the_same_file(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    Bm25LexicalIndex(persist_path).build_index([_chunk("chunk-1", "lockout tagout safety")], chunks_sha256="test")

    fresh_index = Bm25LexicalIndex(persist_path)
    results = fresh_index.query("lockout", top_n=1)

    assert results[0][0] == "chunk-1"


def test_query_raises_file_not_found_with_helpful_message_when_never_built(tmp_path):
    persist_path = tmp_path / "does_not_exist.json"
    index = Bm25LexicalIndex(persist_path)

    try:
        index.query("anything", top_n=1)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "index-build CLI" in str(exc)


def test_validate_passes_when_chunk_ids_match(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    Bm25LexicalIndex(persist_path).build_index(
        [_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta")],
        chunks_sha256="test",
    )
    Bm25LexicalIndex(persist_path).validate(["chunk-1", "chunk-2"])


def test_validate_raises_when_chunk_ids_diverge(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    Bm25LexicalIndex(persist_path).build_index(
        [_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta")],
        chunks_sha256="test",
    )
    import pytest

    with pytest.raises(RuntimeError, match="BM25 chunk ids"):
        Bm25LexicalIndex(persist_path).validate(["chunk-1", "chunk-9"])


def test_no_pickle_import_in_module_source():
    import src.adapters.secondary.lexical.bm25_lexical_index as module

    with open(module.__file__, encoding="utf-8") as f:
        source = f.read()
    assert "pickle" not in source


# --- versioned payload (bucket 3) ---


def test_persisted_payload_is_versioned(tmp_path):
    path = tmp_path / "bm25.json"
    index = Bm25LexicalIndex(path)

    index.build_index([_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta")], chunks_sha256="deadbeef")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == BM25_SCHEMA_VERSION
    assert payload["lexical_profile"] == LEXICAL_PROFILE
    assert payload["chunks_sha256"] == "deadbeef"
    assert payload["chunk_ids"] == ["chunk-1", "chunk-2"]


def test_unversioned_legacy_payload_fails_closed(tmp_path):
    """A pre-versioning index on disk is not silently readable: it was built by
    an unknown tokenizer against an unknown chunk set."""
    path = tmp_path / "bm25.json"
    path.write_text(json.dumps({"chunk_ids": ["a"], "corpus_tokens": [["a"]]}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        Bm25LexicalIndex(path).query("a", 1)


def test_future_schema_version_fails_closed(tmp_path):
    path = tmp_path / "bm25.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": BM25_SCHEMA_VERSION + 1,
                "lexical_profile": LEXICAL_PROFILE,
                "chunks_sha256": "x",
                "chunk_ids": ["a"],
                "corpus_tokens": [["a"]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        Bm25LexicalIndex(path).query("a", 1)


def test_validate_rejects_a_chunks_hash_mismatch(tmp_path):
    """The three artifacts must agree on CONTENT, not just on count: a BM25
    index whose chunks_sha256 differs from the manifest's is scoring a
    different corpus than the vector channel."""
    path = tmp_path / "bm25.json"
    index = Bm25LexicalIndex(path)
    index.build_index([_chunk("chunk-1", "alpha")], chunks_sha256="aaaa")

    with pytest.raises(RuntimeError, match="chunks_sha256"):
        index.validate(["chunk-1"], expected_chunks_sha256="bbbb")


def test_validate_rejects_a_lexical_profile_mismatch(tmp_path):
    path = tmp_path / "bm25.json"
    index = Bm25LexicalIndex(path)
    index.build_index([_chunk("chunk-1", "alpha")], chunks_sha256="aaaa")

    with pytest.raises(RuntimeError, match="lexical_profile"):
        index.validate(
            ["chunk-1"], expected_chunks_sha256="aaaa", expected_lexical_profile="snowball-bilingual-v1"
        )


def test_validate_accepts_a_coherent_index(tmp_path):
    path = tmp_path / "bm25.json"
    index = Bm25LexicalIndex(path)
    index.build_index([_chunk("chunk-1", "alpha")], chunks_sha256="aaaa")

    index.validate(["chunk-1"], expected_chunks_sha256="aaaa")
