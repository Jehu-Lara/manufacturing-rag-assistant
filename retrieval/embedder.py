from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load embedding model '{MODEL_NAME}' — "
                "first run needs network access to download and cache it"
            ) from exc
    return _model


def max_seq_length() -> int:
    return get_model().max_seq_length


def tokenized_length(text: str) -> int:
    """Token count under this model's own tokenizer — NOT tiktoken's cl100k_base
    count used for Phase 1 chunk sizing. The two disagree, which is exactly why
    SPEC.md requires checking the model's own tokenizer before indexing."""
    return len(get_model().tokenizer.encode(text, add_special_tokens=True))


def assert_fits_max_seq_length(texts: list[str]) -> None:
    limit = max_seq_length()
    longest = max((tokenized_length(text) for text in texts), default=0)
    if longest > limit:
        raise ValueError(
            f"a chunk requires {longest} tokens under {MODEL_NAME}'s own tokenizer, "
            f"exceeding its max_seq_length of {limit} — it would be silently truncated"
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = get_model().encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
