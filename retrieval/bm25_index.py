from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from ingestion.metadata import ChunkMetadata

# pickle is safe here: PERSIST_FILE is generated and read only by this module,
# from this project's own gitignored, locally-regenerated build_index output —
# never from an untrusted or externally-supplied source. BM25Okapi has no
# built-in JSON serialization, so pickle is the practical choice for it.
PERSIST_FILE = Path(__file__).resolve().parent / "output" / "bm25_index.pkl"

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Loaded once per process and reused across queries — the persisted index is
# immutable for the lifetime of a running server (rebuilt only by a separate
# `build_index` run, which happens before the server starts, not while it's
# serving traffic), so re-reading and un-pickling it on every /query request
# was pure wasted I/O.
_CACHE: Optional[tuple[list[str], BM25Okapi]] = None


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def build_index(chunks: list[ChunkMetadata]) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    corpus_tokens = [tokenize(chunk.chunk_text) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PERSIST_FILE.open("wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "bm25": bm25}, f)
    global _CACHE
    _CACHE = None


def _load() -> tuple[list[str], BM25Okapi]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not PERSIST_FILE.exists():
        raise FileNotFoundError(f"{PERSIST_FILE} not found — run `python -m retrieval.build_index` first")
    with PERSIST_FILE.open("rb") as f:
        data = pickle.load(f)
    _CACHE = (data["chunk_ids"], data["bm25"])
    return _CACHE


def query(text: str, top_n: int) -> list[tuple[str, float]]:
    """Returns (chunk_id, bm25_score) tuples, best match first."""
    chunk_ids, bm25 = _load()
    scores = bm25.get_scores(tokenize(text))
    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]
