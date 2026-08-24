from __future__ import annotations

from ingestion.loader import CORPUS_ROOT, SOURCES_MANIFEST, load_document, parse_manifest


def test_every_manifest_row_points_to_a_file_that_exists():
    manifest = parse_manifest(SOURCES_MANIFEST)
    assert manifest, "expected at least one row in corpus/SOURCES.md"

    for rel_path, source_type in manifest.items():
        file_path = CORPUS_ROOT / rel_path
        assert file_path.is_file(), f"{rel_path} is listed in SOURCES.md but does not exist on disk"
        assert file_path.parent.name == source_type


def test_every_corpus_file_is_listed_with_a_matching_label():
    manifest = parse_manifest(SOURCES_MANIFEST)
    corpus_files = sorted((CORPUS_ROOT / "public").glob("*.md")) + sorted(
        (CORPUS_ROOT / "synthetic").glob("*.md")
    )
    assert corpus_files, "expected at least one corpus document on disk"

    for file_path in corpus_files:
        rel_path = f"{file_path.parent.name}/{file_path.name}"
        assert rel_path in manifest, f"{rel_path} exists on disk but is not listed in SOURCES.md"

        document = load_document(file_path)
        assert document.source_type == manifest[rel_path] == file_path.parent.name


def test_public_manifest_rows_have_a_real_url():
    manifest = parse_manifest(SOURCES_MANIFEST)

    for rel_path, source_type in manifest.items():
        if source_type != "public":
            continue
        document = load_document(CORPUS_ROOT / rel_path)
        assert document.source_url_or_note.startswith("http"), (
            f"{rel_path}: public document's source_url_or_note should be a real URL, "
            f"got: {document.source_url_or_note!r}"
        )


def test_synthetic_rows_carry_the_fixed_disclosure_note():
    manifest = parse_manifest(SOURCES_MANIFEST)

    for rel_path, source_type in manifest.items():
        if source_type != "synthetic":
            continue
        document = load_document(CORPUS_ROOT / rel_path)
        assert "Synthetic document authored for this portfolio project" in document.source_url_or_note
        assert document.source_page_range is None
