from __future__ import annotations

import json

from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
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

    index.build_index([_chunk("chunk-1", "lockout tagout safety procedure"), _chunk("chunk-2", "quality control unit")])

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
        ]
    )

    results = index.query("lockout tagout", top_n=2)

    assert results[0][0] == "chunk-lockout"


def test_query_loads_from_a_fresh_instance_pointed_at_the_same_file(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    Bm25LexicalIndex(persist_path).build_index([_chunk("chunk-1", "lockout tagout safety")])

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
        [_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta")]
    )
    Bm25LexicalIndex(persist_path).validate(["chunk-1", "chunk-2"])


def test_validate_raises_when_chunk_ids_diverge(tmp_path):
    persist_path = tmp_path / "bm25_index.json"
    Bm25LexicalIndex(persist_path).build_index(
        [_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta")]
    )
    import pytest

    with pytest.raises(RuntimeError, match="BM25 chunk ids"):
        Bm25LexicalIndex(persist_path).validate(["chunk-1", "chunk-9"])


def test_no_pickle_import_in_module_source():
    import src.adapters.secondary.lexical.bm25_lexical_index as module

    with open(module.__file__, encoding="utf-8") as f:
        source = f.read()
    assert "pickle" not in source
