from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from src.domain.models import ChunkMetadata

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

BM25_SCHEMA_VERSION = 1

# Names what tokenize() below actually does. It exists so an index built by a
# different tokenizer (e.g. a Snowball-stemmed experiment) cannot be loaded by
# a runtime expecting this one and silently score a differently-tokenized
# corpus — the failure would otherwise be invisible in the numbers.
LEXICAL_PROFILE = "word-lower-v1"


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
        self._meta: Optional[dict[str, Any]] = None

    def build_index(self, chunks: list[ChunkMetadata], *, chunks_sha256: str) -> None:
        """`chunks_sha256` is required, not optional: it is what lets startup
        prove this index was built from the same chunks.jsonl the manifest and
        the vector collection describe. The write stays atomic (tmp+replace),
        so a crash mid-write leaves the previous index intact; the build CLI
        writes the manifest afterwards as the commit marker."""
        payload = {
            "schema_version": BM25_SCHEMA_VERSION,
            "lexical_profile": LEXICAL_PROFILE,
            "chunks_sha256": chunks_sha256,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "corpus_tokens": [tokenize(chunk.chunk_text) for chunk in chunks],
        }

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self._persist_path)

        self._chunk_ids = None
        self._bm25 = None
        self._meta = None

    def _load(self) -> tuple[list[str], BM25Okapi]:
        if self._chunk_ids is not None and self._bm25 is not None:
            return self._chunk_ids, self._bm25
        if not self._persist_path.exists():
            raise FileNotFoundError(f"{self._persist_path} not found — run the retrieval index-build CLI first")
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != BM25_SCHEMA_VERSION:
            raise ValueError(
                f"{self._persist_path} has schema_version {version!r}, expected {BM25_SCHEMA_VERSION} — "
                "rebuild the index (`python -m src.features.retrieval.cli`)"
            )
        self._meta = {
            "lexical_profile": data.get("lexical_profile"),
            "chunks_sha256": data.get("chunks_sha256"),
        }
        self._chunk_ids = data["chunk_ids"]
        self._bm25 = BM25Okapi(data["corpus_tokens"])
        return self._chunk_ids, self._bm25

    def validate(
        self,
        expected_chunk_ids: list[str],
        *,
        expected_chunks_sha256: Optional[str] = None,
        expected_lexical_profile: str = LEXICAL_PROFILE,
    ) -> None:
        """Startup guard: the persisted BM25 corpus must cover exactly the
        indexed chunks, in the same order, AND have been built from the same
        chunks.jsonl by the same tokenizer — otherwise the lexical channel is
        scoring a different corpus than the vector channel, which shows up in
        the numbers as nothing at all."""
        chunk_ids, _ = self._load()
        meta = self._meta or {}
        if chunk_ids != expected_chunk_ids:
            raise RuntimeError(
                f"BM25 chunk ids do not match the indexed chunks "
                f"({len(chunk_ids)} persisted vs {len(expected_chunk_ids)} expected)"
            )
        actual_profile = meta.get("lexical_profile")
        if actual_profile != expected_lexical_profile:
            raise RuntimeError(
                f"BM25 lexical_profile is {actual_profile!r}, expected {expected_lexical_profile!r}"
            )
        actual_digest = meta.get("chunks_sha256")
        if expected_chunks_sha256 is not None and actual_digest != expected_chunks_sha256:
            raise RuntimeError(
                f"BM25 chunks_sha256 is {actual_digest!r}, expected {expected_chunks_sha256!r} — "
                "the lexical index was built from a different chunks.jsonl"
            )

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        """Returns (chunk_id, bm25_score) tuples, best match first."""
        chunk_ids, bm25 = self._load()
        scores = bm25.get_scores(tokenize(text))
        ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
