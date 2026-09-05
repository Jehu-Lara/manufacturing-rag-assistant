"""Cross-encoder reranker (audit bucket 5, task 3). Opt-in, never a default.

A bi-encoder scores query and passage independently; a cross-encoder reads both
together and is materially better at ordering a short candidate list. The cost
is that it cannot be precomputed — every query pays a forward pass per
candidate — so this is wired only where a caller asks for it explicitly.

`CrossEncoder` ships with `sentence-transformers`, already a runtime
dependency, so nothing new is installed. What is NOT free is the model: about
2.3GB of weights downloaded on first use, plus per-query latency. That makes
enabling this a deploy decision with its own latency budget, which is why
`HybridRetriever` defaults to `reranker=None` and the composition root
constructs none.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sentence_transformers import CrossEncoder

from src.core.telemetry import get_tracer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class FlagReranker:
    """Implements RerankerPort. Model loaded lazily on first rerank() and held
    on the instance, mirroring SentenceTransformersEmbedder — no module-level
    global, so the composition root owns its lifetime."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load reranker model '{self._model_name}' — "
                    "first run needs network access to download and cache it"
                ) from exc
        return self._model

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """Returns the SAME id set, best-first. The retriever rejects anything
        else, because the refusal gate's guarantee rests on every fused result
        still being present after reranking."""
        if not candidates:
            return []
        with get_tracer().start_as_current_span("reranker.score"):
            scores = self._get_model().predict([(query, text) for _, text in candidates])
            scored = list(scores)
            if len(scored) != len(candidates):
                raise ValueError(
                    f"reranker returned {len(scored)} scores for {len(candidates)} candidates — "
                    "expected one score per candidate"
                )
            ranked = [(chunk_id, float(score)) for (chunk_id, _), score in zip(candidates, scored)]
            ranked.sort(key=lambda pair: (-pair[1], pair[0]))
            return ranked
