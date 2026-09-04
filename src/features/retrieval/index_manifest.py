from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION
from src.core.config import Settings, load_settings
from src.core.paths import CHUNKS_FILE, CORPUS_DIR, REPO_ROOT, RETRIEVAL_OUTPUT_DIR
from src.domain.models import IndexProfile

MANIFEST_FILE = RETRIEVAL_OUTPUT_DIR / "index_manifest.json"

_SHA1_RE = re.compile(r"[0-9a-fA-F]{40}")

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


def resolve_index_profile(settings: Settings | None = None) -> IndexProfile:
    """The INDEX_PROFILE env read lives in load_settings(); this keeps the
    zero-arg call convenience while leaving one authority for the variable.
    No cast is needed: config's IndexProfileName and domain's IndexProfile are
    the same Literal, pinned equal by tests/test_core_config.py."""
    resolved = settings if settings is not None else load_settings()
    return resolved.index_profile


def resolve_build_commit(explicit: str | None = None, *, settings: Settings | None = None) -> str:
    if explicit:
        return explicit
    resolved = settings if settings is not None else load_settings()
    if resolved.deployed_sha:
        return resolved.deployed_sha
    deployed_file = REPO_ROOT / "DEPLOYED_SHA"
    if deployed_file.exists():
        value = deployed_file.read_text(encoding="utf-8").strip()
        if _SHA1_RE.fullmatch(value):
            return value
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


def _chunk_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def verify(
    path: Path = MANIFEST_FILE,
    *,
    expected_profile: IndexProfile | None = None,
    chunks_path: Path = CHUNKS_FILE,
    corpus_dir: Path = CORPUS_DIR,
) -> IndexManifest:
    manifest = read(path)
    actual_chunks = chunks_sha256(chunks_path)
    actual_corpus = corpus_sha256(corpus_dir)
    actual_chunk_count = _chunk_line_count(chunks_path)
    mismatches: list[str] = []
    if manifest.chunks_sha256 != actual_chunks:
        mismatches.append(f"chunks_sha256 stored {manifest.chunks_sha256}, computed {actual_chunks}")
    if manifest.corpus_sha256 != actual_corpus:
        mismatches.append(f"corpus_sha256 stored {manifest.corpus_sha256}, computed {actual_corpus}")
    if manifest.embedding_model != MODEL_NAME:
        mismatches.append(f"embedding_model stored {manifest.embedding_model}, expected {MODEL_NAME}")
    if manifest.embedding_revision != MODEL_REVISION:
        mismatches.append(
            f"embedding_revision stored {manifest.embedding_revision}, expected {MODEL_REVISION}"
        )
    if manifest.chunk_count != actual_chunk_count:
        mismatches.append(
            f"chunk_count stored {manifest.chunk_count}, computed {actual_chunk_count} from {chunks_path.name}"
        )
    if expected_profile is not None and manifest.index_profile != expected_profile:
        mismatches.append(
            f"index_profile stored {manifest.index_profile!r}, expected {expected_profile!r}"
        )
    if mismatches:
        raise ValueError(
            f"{path} no longer matches the current inputs — "
            + "; ".join(mismatches)
            + ". Rebuild the index (`python -m src.features.retrieval.cli`)."
        )
    return manifest
