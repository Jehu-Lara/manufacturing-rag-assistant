from __future__ import annotations

from unittest.mock import patch

import pytest

from src.adapters.secondary.reranker.flag_reranker import MODEL_NAME, FlagReranker
from src.domain.ports import RerankerPort

CANDIDATES = [("c01", "first text"), ("c02", "second text"), ("c03", "third text")]


def test_adapter_satisfies_the_port() -> None:
    assert isinstance(FlagReranker(), RerankerPort)


def test_model_is_not_loaded_at_construction() -> None:
    """~2.3GB of weights must not download because someone imported the module
    or built the object. Loading is lazy, on the first rerank()."""
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder") as cross_encoder:
        FlagReranker()

    cross_encoder.assert_not_called()


def test_reranks_best_first_over_the_same_id_set() -> None:
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder") as cross_encoder:
        cross_encoder.return_value.predict.return_value = [0.1, 0.9, 0.5]

        ranked = FlagReranker().rerank("query", CANDIDATES)

    assert [chunk_id for chunk_id, _ in ranked] == ["c02", "c03", "c01"]
    assert {chunk_id for chunk_id, _ in ranked} == {"c01", "c02", "c03"}


def test_pairs_the_query_with_every_candidate_text() -> None:
    """A cross-encoder scores (query, passage) pairs. Handing it chunk_ids
    instead of chunk text would produce plausible-looking noise."""
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder") as cross_encoder:
        cross_encoder.return_value.predict.return_value = [0.1, 0.2, 0.3]

        FlagReranker().rerank("what is the PEL?", CANDIDATES)

    pairs = cross_encoder.return_value.predict.call_args.args[0]
    assert pairs == [("what is the PEL?", text) for _, text in CANDIDATES]


def test_model_is_loaded_once_and_reused() -> None:
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder") as cross_encoder:
        cross_encoder.return_value.predict.return_value = [0.1, 0.2, 0.3]
        reranker = FlagReranker()
        reranker.rerank("q", CANDIDATES)
        reranker.rerank("q", CANDIDATES)

    cross_encoder.assert_called_once_with(MODEL_NAME)


def test_empty_candidates_never_touch_the_model() -> None:
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder") as cross_encoder:
        assert FlagReranker().rerank("q", []) == []

    cross_encoder.assert_not_called()


def test_a_load_failure_says_what_is_missing() -> None:
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder", side_effect=OSError("no net")):
        with pytest.raises(RuntimeError, match=MODEL_NAME):
            FlagReranker().rerank("q", CANDIDATES)


def test_a_score_count_mismatch_fails_closed() -> None:
    """Silently zipping a short score list would drop candidates, which is
    exactly the id-set violation HybridRetriever refuses to accept."""
    with patch("src.adapters.secondary.reranker.flag_reranker.CrossEncoder") as cross_encoder:
        cross_encoder.return_value.predict.return_value = [0.1, 0.2]

        with pytest.raises(ValueError, match="one score per candidate"):
            FlagReranker().rerank("q", CANDIDATES)


def test_the_adapter_is_not_wired_into_serving() -> None:
    """Opt-in only. The composition root must not construct one, because doing
    so adds a cross-encoder pass to every served query and ~2.3GB to the image
    — a deploy decision with its own latency budget, not a default."""
    from pathlib import Path

    for module in (Path("src/main.py"), Path("src/adapters/primary/http/app.py")):
        assert "FlagReranker" not in module.read_text(encoding="utf-8"), module
