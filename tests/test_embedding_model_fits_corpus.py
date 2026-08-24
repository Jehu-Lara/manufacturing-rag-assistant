from __future__ import annotations

from retrieval.build_index import load_chunks
from retrieval.embedder import assert_fits_max_seq_length, max_seq_length, tokenized_length


def test_model_max_seq_length_covers_every_real_corpus_chunk():
    # Guards the exact trap SPEC.md's Language Policy warns about: a model
    # disqualified by its OWN tokenizer's max_seq_length, checked against
    # real chunks — not against tiktoken's cl100k_base count (which disagrees
    # with this model's tokenizer and is not what would actually truncate).
    chunks = load_chunks()
    assert_fits_max_seq_length([chunk.chunk_text for chunk in chunks])


def test_tokenized_length_is_positive_for_nonempty_text():
    assert tokenized_length("hello world") > 0


def test_max_seq_length_is_a_positive_integer():
    assert isinstance(max_seq_length(), int)
    assert max_seq_length() > 0
