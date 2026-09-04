from __future__ import annotations

import ast
import json
from pathlib import Path

from src.domain.models import ChunkMetadata
from src.features.retrieval.chunk_store import CHUNKS_FILE, load_chunks

MODULE = Path(__file__).resolve().parent.parent / "src" / "features" / "retrieval" / "chunk_store.py"


def _chunk_row() -> dict[str, object]:
    """Built against ChunkMetadata's real field list so a field rename fails
    here loudly instead of drifting into a TypeError inside the behaviour
    tests below."""
    defaults: dict[str, object] = {
        "chunk_id": "doc-a::chunk-0000",
        "document_id": "doc-a",
        "document_title": "Doc A",
        "revision": "1.0",
        "section_heading": "S1",
        "source_type": "synthetic",
        "source_url_or_note": "note",
        "source_page_range": None,
        "md_line_range": "1-2",
        "chunk_token_count": 2,
        "chunk_text": "body",
    }
    fields = {f for f in ChunkMetadata.__dataclass_fields__}
    assert set(defaults) == fields, (
        f"update _chunk_row: missing={sorted(fields - set(defaults))} "
        f"stale={sorted(set(defaults) - fields)}"
    )
    return defaults


def test_chunk_store_imports_no_adapter() -> None:
    """Serving reads chunk ids at startup; it must not have to import the
    index-build CLI (and through it chromadb + sentence-transformers) to do it."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    forbidden = {
        m
        for m in modules
        if m.startswith("src.adapters")
        or m.split(".")[0] in {"chromadb", "sentence_transformers", "torch"}
    }
    assert not forbidden, f"chunk_store must stay adapter-free, imported: {forbidden}"


def test_load_chunks_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(_chunk_row()) + "\n", encoding="utf-8")

    chunks = load_chunks(path)

    assert [c.chunk_id for c in chunks] == ["doc-a::chunk-0000"]
    assert chunks[0].source_page_range is None


def test_load_chunks_missing_file_names_the_ingestion_command(tmp_path: Path) -> None:
    try:
        load_chunks(tmp_path / "absent.jsonl")
    except FileNotFoundError as exc:
        assert "src.features.ingestion.cli" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_default_chunks_file_is_the_repo_ingestion_output() -> None:
    assert CHUNKS_FILE.parts[-3:] == ("ingestion", "output", "chunks.jsonl")


def test_cli_still_re_exports_load_chunks() -> None:
    """Compatibility: `from src.features.retrieval.cli import load_chunks` was
    the only import path before this split."""
    from src.features.retrieval.cli import CHUNKS_FILE as cli_chunks_file
    from src.features.retrieval.cli import load_chunks as cli_load_chunks

    assert cli_load_chunks is load_chunks
    assert cli_chunks_file == CHUNKS_FILE
