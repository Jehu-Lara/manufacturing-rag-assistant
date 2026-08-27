from __future__ import annotations

from src.features.ingestion.chunker import TiktokenCounter
from src.features.ingestion.cli import build_chunks_for_document
from src.features.ingestion.use_cases import load_corpus

_REQUIRED_STRING_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "revision",
    "section_heading",
    "source_type",
    "source_url_or_note",
    "md_line_range",
    "chunk_text",
)


def _all_chunks():
    counter = TiktokenCounter()
    documents = load_corpus()
    chunks = []
    for document in documents:
        chunks.extend(build_chunks_for_document(document, counter))
    return documents, chunks


def _all_chunks_with_documents():
    counter = TiktokenCounter()
    documents = load_corpus()
    pairs = []
    for document in documents:
        for chunk in build_chunks_for_document(document, counter):
            pairs.append((document, chunk))
    return pairs


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
    counter = TiktokenCounter()
    synthetic_docs = [d for d in documents if d.source_type == "synthetic"]
    assert synthetic_docs, "expected at least one synthetic document in the corpus"

    for document in synthetic_docs:
        for chunk in build_chunks_for_document(document, counter):
            # Synthetic documents intentionally have no page pagination — None
            # is the expected, valid value here, not a missing-field failure.
            assert chunk.source_page_range is None

    public_docs = [d for d in documents if d.source_type == "public"]
    assert public_docs, "expected at least one public document in the corpus"
    for document in public_docs:
        for chunk in build_chunks_for_document(document, counter):
            assert chunk.source_page_range is None or isinstance(chunk.source_page_range, str)


def test_every_chunk_id_is_unique():
    _, chunks = _all_chunks()
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_md_line_range_actually_brackets_the_chunk_text_in_the_source_file():
    # Regression test: an earlier version of the chunker trimmed leading/
    # trailing blank lines from a section's text without adjusting
    # start_line/end_line to match, so md_line_range silently pointed short
    # of the true content. This checks the file's actual line at
    # md_line_range's start/end matches chunk_text's first/last line
    # exactly, for every chunk in the real corpus.
    for document, chunk in _all_chunks_with_documents():
        file_lines = document.file_path.read_text(encoding="utf-8").splitlines()
        start, end = (int(x) for x in chunk.md_line_range.split("-"))

        chunk_lines = chunk.chunk_text.split("\n")
        assert file_lines[start - 1] == chunk_lines[0], (
            f"{chunk.chunk_id}: md_line_range start ({start}) doesn't match chunk_text's first line"
        )
        assert file_lines[end - 1] == chunk_lines[-1], (
            f"{chunk.chunk_id}: md_line_range end ({end}) doesn't match chunk_text's last line"
        )
