"""Experiment-only bilingual BM25 (audit bucket 5).

NOT part of the served runtime: `snowballstemmer` lives in
`requirements-experiments.txt`, never in `requirements-lock.txt`, so importing
this module fails on a production install — deliberately. The ablation harness
catches that ImportError and says which extra to install.

Whether stemming beats the shipped `word-lower-v1` tokenizer is the hypothesis
the ablation measures. It is not a claim, and nothing here changes a default.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import snowballstemmer
from rank_bm25 import BM25Okapi

from src.domain.models import ChunkMetadata

BM25_SCHEMA_VERSION = 1
LEXICAL_PROFILE = "snowball-bilingual-v1"

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# A tuple of stateless stemmer objects built once at import. This is NOT the
# module-level mutable singleton CLAUDE.md prohibits: it is immutable and holds
# no per-call state, so there is nothing to go stale or leak between callers.
_STEMMERS = tuple(snowballstemmer.stemmer(language) for language in ("english", "spanish"))


def bilingual_tokenize(text: str) -> list[str]:
    """Emits BOTH language stems per token, deliberately. The corpus is
    bilingual and a chunk's language is not known at query time, so applying a
    single language's stemmer would degrade the other half — which is the most
    likely way a stemmed index ends up worse than no stemming at all. Duplicate
    stems (the common case, where both stemmers agree) are collapsed so term
    frequencies stay honest."""
    tokens: list[str] = []
    for raw in _TOKEN_PATTERN.findall(text.lower()):
        seen: list[str] = []
        for stemmer in _STEMMERS:
            stem = stemmer.stemWord(raw)
            if stem not in seen:
                seen.append(stem)
        tokens.extend(seen)
    return tokens


class SnowballBm25Index:
    """Implements LexicalIndexPort. Mirrors Bm25LexicalIndex's versioned
    payload but stamps its own LEXICAL_PROFILE, so a word-lower-v1 runtime
    refuses to load it rather than silently scoring a differently-tokenized
    corpus. Deliberately a separate class rather than a configurable
    Bm25LexicalIndex: a little duplication in an experiment-only adapter is
    cheaper than making the production class pluggable for a hypothesis that
    may not survive."""

    def __init__(self, persist_path: Path) -> None:
        self._persist_path = persist_path
        self._chunk_ids: Optional[list[str]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._meta: Optional[dict[str, Any]] = None

    def build_index(self, chunks: list[ChunkMetadata], *, chunks_sha256: str) -> None:
        payload = {
            "schema_version": BM25_SCHEMA_VERSION,
            "lexical_profile": LEXICAL_PROFILE,
            "chunks_sha256": chunks_sha256,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "corpus_tokens": [bilingual_tokenize(chunk.chunk_text) for chunk in chunks],
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
            raise FileNotFoundError(
                f"{self._persist_path} not found — build it with "
                "`python -m src.features.evaluation.ablation_eval --build-snowball`"
            )
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != BM25_SCHEMA_VERSION:
            raise ValueError(
                f"{self._persist_path} has schema_version {data.get('schema_version')!r}, "
                f"expected {BM25_SCHEMA_VERSION}"
            )
        if data.get("lexical_profile") != LEXICAL_PROFILE:
            raise ValueError(
                f"{self._persist_path} has lexical_profile {data.get('lexical_profile')!r}, "
                f"expected {LEXICAL_PROFILE}"
            )
        self._meta = {"chunks_sha256": data.get("chunks_sha256")}
        self._chunk_ids = data["chunk_ids"]
        self._bm25 = BM25Okapi(data["corpus_tokens"])
        return self._chunk_ids, self._bm25

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        chunk_ids, bm25 = self._load()
        scores = bm25.get_scores(bilingual_tokenize(text))
        ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
