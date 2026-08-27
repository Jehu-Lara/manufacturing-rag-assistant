from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from src.domain.models import SourceType

CORPUS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "corpus"
SOURCES_MANIFEST = CORPUS_ROOT / "SOURCES.md"

_FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)
_MANIFEST_ROW_PATTERN = re.compile(r"\|\s*`((?:public|synthetic)/[^`]+\.md)`\s*\|")

_REQUIRED_FRONTMATTER_KEYS = (
    "document_id",
    "document_title",
    "revision",
    "source_type",
    "source_url_or_note",
)


@dataclass(frozen=True)
class Document:
    document_id: str
    document_title: str
    revision: str
    source_type: SourceType
    source_url_or_note: str
    source_page_range: Optional[str]
    file_path: Path
    body: str
    body_start_line: int
    """1-indexed line number, in the full source file, where `body` begins."""


def _parse_frontmatter(text: str, file_path: Path) -> tuple[dict[str, Any], str, int]:
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        raise ValueError(f"{file_path}: missing YAML frontmatter block (expected '---' ... '---')")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{file_path}: frontmatter did not parse to a mapping")
    for key in _REQUIRED_FRONTMATTER_KEYS:
        if not frontmatter.get(key):
            raise ValueError(f"{file_path}: frontmatter missing required key '{key}'")
    if frontmatter["source_type"] not in ("public", "synthetic"):
        raise ValueError(f"{file_path}: invalid source_type '{frontmatter['source_type']}'")
    body = text[match.end():]
    body_start_line = match.group(0).count("\n") + 1
    return frontmatter, body, body_start_line


def load_document(file_path: Path) -> Document:
    text = file_path.read_text(encoding="utf-8")
    frontmatter, body, body_start_line = _parse_frontmatter(text, file_path)

    expected_type = "public" if file_path.parent.name == "public" else "synthetic"
    if frontmatter["source_type"] != expected_type:
        raise ValueError(
            f"{file_path}: frontmatter source_type '{frontmatter['source_type']}' "
            f"does not match its directory (expected '{expected_type}')"
        )
    source_type: SourceType = frontmatter["source_type"]  # validated against ("public", "synthetic") above

    return Document(
        document_id=frontmatter["document_id"],
        document_title=frontmatter["document_title"],
        revision=frontmatter["revision"],
        source_type=source_type,
        source_url_or_note=frontmatter["source_url_or_note"],
        source_page_range=frontmatter.get("source_page_range"),
        file_path=file_path,
        body=body,
        body_start_line=body_start_line,
    )


def parse_manifest(manifest_path: Path = SOURCES_MANIFEST) -> dict[str, str]:
    """Return {"public/foo.md": "public", "synthetic/bar.md": "synthetic", ...} from SOURCES.md's tables."""
    text = manifest_path.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = _MANIFEST_ROW_PATTERN.search(line)
        if match:
            rel_path = match.group(1)
            entries[rel_path] = rel_path.split("/", 1)[0]
    return entries


def _validate_against_manifest(documents: Iterable[Document], manifest_path: Path) -> None:
    manifest = parse_manifest(manifest_path)
    for doc in documents:
        rel_path = f"{doc.file_path.parent.name}/{doc.file_path.name}"
        if rel_path not in manifest:
            raise ValueError(f"{doc.file_path}: not listed in {manifest_path}")
        if manifest[rel_path] != doc.source_type:
            raise ValueError(
                f"{doc.file_path}: {manifest_path} lists it as '{manifest[rel_path]}' "
                f"but its frontmatter says '{doc.source_type}'"
            )


def load_corpus(corpus_root: Path = CORPUS_ROOT) -> list[Document]:
    md_files = sorted((corpus_root / "public").glob("*.md")) + sorted(
        (corpus_root / "synthetic").glob("*.md")
    )
    if not md_files:
        raise ValueError(f"no corpus documents found under {corpus_root}")

    documents = [load_document(path) for path in md_files]
    _validate_against_manifest(documents, corpus_root / "SOURCES.md")
    return documents
