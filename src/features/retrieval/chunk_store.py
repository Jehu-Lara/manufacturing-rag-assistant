from __future__ import annotations

import json
from pathlib import Path

from src.core.paths import CHUNKS_FILE
from src.domain.models import ChunkMetadata

__all__ = ["CHUNKS_FILE", "load_chunks"]


def load_chunks(path: Path = CHUNKS_FILE) -> list[ChunkMetadata]:
    """Deliberately adapter-free: serving reads chunk ids at startup and must
    not import the index-build CLI (and through it chromadb and
    sentence-transformers) to do it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.features.ingestion.cli` first to produce it"
        )
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(ChunkMetadata(**json.loads(line)))
    return chunks
