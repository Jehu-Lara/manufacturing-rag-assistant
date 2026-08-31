from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHUNKS_FILE = REPO_ROOT / "ingestion" / "output" / "chunks.jsonl"
CORPUS_DIR = REPO_ROOT / "corpus"
MANIFEST_FILE = REPO_ROOT / "retrieval" / "output" / "index_manifest.json"

_MANIFEST_FIELDS = (
    "index_profile",
    "chunks_sha256",
    "corpus_sha256",
    "embedding_model",
    "embedding_revision",
    "build_commit",
    "chunk_count",
)


def chunks_sha256(path: Path = CHUNKS_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_sha256(corpus_dir: Path = CORPUS_DIR) -> str:
    """Hash over the exact set of files ingestion embeds — ``public/*.md`` and
    ``synthetic/*.md`` only (mirrors ``ingestion.use_cases.load_corpus``) — as
    sorted relative POSIX paths each followed by its file bytes. Renaming an
    embedded file changes the hash; ``corpus/SOURCES.md``, other stray ``.md``,
    ``.env``, PDFs, generated output, and mtimes never contribute."""
    digest = hashlib.sha256()
    md_files = sorted(
        [*(corpus_dir / "public").glob("*.md"), *(corpus_dir / "synthetic").glob("*.md")],
        key=lambda p: p.relative_to(corpus_dir).as_posix(),
    )
    for md_file in md_files:
        rel_posix = md_file.relative_to(corpus_dir).as_posix()
        digest.update(rel_posix.encode("utf-8"))
        digest.update(b"\0")
        digest.update(md_file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_build_commit(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    deployed_sha = os.environ.get("DEPLOYED_SHA")
    if deployed_sha:
        return deployed_sha
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
    return head or "unknown"


@dataclass(frozen=True)
class IndexManifest:
    index_profile: str
    chunks_sha256: str
    corpus_sha256: str
    embedding_model: str
    embedding_revision: str
    build_commit: str
    chunk_count: int


def build_manifest(
    index_profile: str,
    chunk_count: int,
    *,
    build_commit: str | None = None,
    chunks_path: Path = CHUNKS_FILE,
    corpus_dir: Path = CORPUS_DIR,
) -> IndexManifest:
    return IndexManifest(
        index_profile=index_profile,
        chunks_sha256=chunks_sha256(chunks_path),
        corpus_sha256=corpus_sha256(corpus_dir),
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        build_commit=resolve_build_commit(build_commit),
        chunk_count=chunk_count,
    )


def write(manifest: IndexManifest, path: Path = MANIFEST_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read(path: Path = MANIFEST_FILE) -> IndexManifest:
    with path.open("r", encoding="utf-8") as f:
        data = cast("dict[str, Any]", json.load(f))
    missing = [field for field in _MANIFEST_FIELDS if field not in data]
    if missing:
        raise ValueError(f"{path} is missing manifest fields: {missing}")
    if data["index_profile"] not in ("raw-v1", "contextual-v1"):
        raise ValueError(f"{path} has an invalid index_profile: {data['index_profile']!r}")
    if not isinstance(data["chunk_count"], int) or isinstance(data["chunk_count"], bool):
        raise ValueError(f"{path} has a non-int chunk_count: {data['chunk_count']!r}")
    return IndexManifest(**{field: data[field] for field in _MANIFEST_FIELDS})


def verify(
    path: Path = MANIFEST_FILE,
    *,
    chunks_path: Path = CHUNKS_FILE,
    corpus_dir: Path = CORPUS_DIR,
) -> None:
    manifest = read(path)
    actual_chunks = chunks_sha256(chunks_path)
    actual_corpus = corpus_sha256(corpus_dir)
    mismatches: list[str] = []
    if manifest.chunks_sha256 != actual_chunks:
        mismatches.append(f"chunks_sha256 stored {manifest.chunks_sha256}, computed {actual_chunks}")
    if manifest.corpus_sha256 != actual_corpus:
        mismatches.append(f"corpus_sha256 stored {manifest.corpus_sha256}, computed {actual_corpus}")
    if mismatches:
        raise ValueError(
            f"{path} no longer matches the current inputs — "
            + "; ".join(mismatches)
            + ". Rebuild the index (`python -m src.features.retrieval.cli`)."
        )
