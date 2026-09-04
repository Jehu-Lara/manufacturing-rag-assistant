from __future__ import annotations

from typing import Any

from src.domain.models import ChunkMetadata


class NullLexicalIndex:
    """Implements LexicalIndexPort with no lexical signal at all.

    Exists so the ablation's semantic-only arm runs the REAL RRF fusion against
    an empty BM25 ranking, rather than a branch inside HybridRetriever that
    skips fusion. Measuring a code path production never takes would answer the
    wrong question, and putting an experiment-only `if` on the serving path is
    exactly what this avoids."""

    def build_index(self, chunks: list[ChunkMetadata], **kwargs: Any) -> None:
        return None

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        return []
