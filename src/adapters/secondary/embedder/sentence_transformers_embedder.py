from __future__ import annotations

from typing import Optional, cast

from sentence_transformers import SentenceTransformer

from src.core.telemetry import get_tracer

MODEL_NAME = "BAAI/bge-m3"


class SentenceTransformersEmbedder:
    """Implements EmbedderPort. Model is loaded lazily and cached on the
    instance (not a module-level global) — one instance is constructed once
    in the composition root and reused for the process lifetime."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load embedding model '{self._model_name}' — "
                    "first run needs network access to download and cache it"
                ) from exc
        return self._model

    def max_seq_length(self) -> int:
        value = self._get_model().max_seq_length
        assert value is not None, f"{self._model_name} did not report a max_seq_length"
        return value

    def tokenized_length(self, text: str) -> int:
        """Token count under this model's own tokenizer — NOT tiktoken's
        cl100k_base count used for chunk sizing. The two disagree, which is
        exactly why the model's own tokenizer must be checked before indexing."""
        return len(self._get_model().tokenizer.encode(text, add_special_tokens=True))

    def assert_fits_max_seq_length(self, texts: list[str]) -> None:
        limit = self.max_seq_length()
        longest = max((self.tokenized_length(text) for text in texts), default=0)
        if longest > limit:
            raise ValueError(
                f"a chunk requires {longest} tokens under {self._model_name}'s own tokenizer, "
                f"exceeding its max_seq_length of {limit} — it would be silently truncated"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        with get_tracer().start_as_current_span("embedder.compute"):
            embeddings = self._get_model().encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return cast("list[list[float]]", embeddings.tolist())

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
