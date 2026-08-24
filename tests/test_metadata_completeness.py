from __future__ import annotations

from ingestion.loader import load_corpus
from ingestion.run import build_chunks_for_document

_REQUIRED_STRING_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "revision",
    "section_heading",
    "source_type",
    "source_url_or_note",
    "md_line_range",
)


def _all_chunks():
    documents = load_corpus()
    chunks = []
    for document in documents:
        chunks.extend(build_chunks_for_document(document))
    return documents, chunks


def test_every_chunk_has_complete_required_metadata():
    documents, chunks = _all_chunks()
    assert documents, "expected at least one corpus document"
    assert chunks, "expected at least one chunk to be produced"

    for chunk in chunks:
        for field_name in _REQUIRED_STRING_FIELDS:
            value = getattr(chunk, field_name)
            assert value and str(value).strip(), f"{chunk.chunk_id}: missing required field '{field_name}'"
        assert chunk.source_type in ("public", "synthetic")
        assert chunk.chunk_token_count > 0


def test_source_page_range_is_optional_not_required():
    documents, _ = _all_chunks()
    synthetic_docs = [d for d in documents if d.source_type == "synthetic"]
    assert synthetic_docs, "expected at least one synthetic document in the corpus"

    for document in synthetic_docs:
        for chunk in build_chunks_for_document(document):
            # Synthetic documents intentionally have no page pagination — None
            # is the expected, valid value here, not a missing-field failure.
            assert chunk.source_page_range is None

    public_docs = [d for d in documents if d.source_type == "public"]
    assert public_docs, "expected at least one public document in the corpus"
    for document in public_docs:
        for chunk in build_chunks_for_document(document):
            assert chunk.source_page_range is None or isinstance(chunk.source_page_range, str)


def test_every_chunk_id_is_unique():
    _, chunks = _all_chunks()
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
