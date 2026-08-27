from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.features.retrieval.cli import load_chunks


def test_model_max_seq_length_covers_every_real_corpus_chunk():
    # Guards a model disqualified by its OWN tokenizer's max_seq_length,
    # checked against real chunks — not against tiktoken's cl100k_base count
    # (which disagrees with this model's tokenizer and is not what would
    # actually truncate).
    chunks = load_chunks()
    embedder = SentenceTransformersEmbedder()
    embedder.assert_fits_max_seq_length([chunk.chunk_text for chunk in chunks])


def test_tokenized_length_is_positive_for_nonempty_text():
    embedder = SentenceTransformersEmbedder()
    assert embedder.tokenized_length("hello world") > 0


def test_max_seq_length_is_a_positive_integer():
    embedder = SentenceTransformersEmbedder()
    assert isinstance(embedder.max_seq_length(), int)
    assert embedder.max_seq_length() > 0
