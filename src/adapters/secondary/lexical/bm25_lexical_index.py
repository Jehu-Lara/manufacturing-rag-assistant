from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from src.domain.models import ChunkMetadata

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class Bm25LexicalIndex:
    """Implements LexicalIndexPort. Persists the tokenized corpus as JSON and
    rebuilds BM25Okapi in memory on load — not the fitted model itself
    (BM25Okapi has no native serialization, and pickling it is exactly the
    runtime-code smell this move eliminates; see ADR-004). Rebuilding from
    tokens is a single term-frequency pass, cheap enough to pay once per
    process start."""

    def __init__(self, persist_path: Path) -> None:
        self._persist_path = persist_path
        self._chunk_ids: Optional[list[str]] = None
        self._bm25: Optional[BM25Okapi] = None

    def build_index(self, chunks: list[ChunkMetadata]) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        corpus_tokens = [tokenize(chunk.chunk_text) for chunk in chunks]
        payload = {"chunk_ids": chunk_ids, "corpus_tokens": corpus_tokens}

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self._persist_path)

        self._chunk_ids = None
        self._bm25 = None

    def _load(self) -> tuple[list[str], BM25Okapi]:
        if self._chunk_ids is not None and self._bm25 is not None:
            return self._chunk_ids, self._bm25
        if not self._persist_path.exists():
            raise FileNotFoundError(f"{self._persist_path} not found — run the retrieval index-build CLI first")
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        self._chunk_ids = data["chunk_ids"]
        self._bm25 = BM25Okapi(data["corpus_tokens"])
        return self._chunk_ids, self._bm25

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        """Returns (chunk_id, bm25_score) tuples, best match first."""
        chunk_ids, bm25 = self._load()
        scores = bm25.get_scores(tokenize(text))
        ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
