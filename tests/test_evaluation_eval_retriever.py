from __future__ import annotations

import pytest

from src.features.evaluation._eval_retriever import assert_live_index_profile
from src.features.retrieval import index_manifest


def _fake_manifest(profile: str) -> index_manifest.IndexManifest:
    return index_manifest.IndexManifest(
        index_profile=profile,
        chunks_sha256="x",
        corpus_sha256="y",
        embedding_model="m",
        embedding_revision="r",
        build_commit="c",
        chunk_count=1,
    )


def test_raises_when_live_profile_differs_from_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_manifest, "read", lambda: _fake_manifest("raw-v1"))
    with pytest.raises(RuntimeError, match="contextual-v1"):
        assert_live_index_profile("contextual-v1")


def test_raises_mentioning_the_cli_when_manifest_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> index_manifest.IndexManifest:
        raise FileNotFoundError

    monkeypatch.setattr(index_manifest, "read", _missing)
    with pytest.raises(RuntimeError, match="python -m src.features.retrieval.cli"):
        assert_live_index_profile("contextual-v1")


def test_returns_none_when_profiles_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_manifest, "read", lambda: _fake_manifest("raw-v1"))
    monkeypatch.setattr(index_manifest, "verify", lambda: None)
    assert assert_live_index_profile("raw-v1") is None


def test_propagates_value_error_when_profile_matches_but_hashes_drifted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(index_manifest, "read", lambda: _fake_manifest("raw-v1"))

    def _drifted() -> None:
        raise ValueError("chunks_sha256 stored x, computed y")

    monkeypatch.setattr(index_manifest, "verify", _drifted)
    with pytest.raises(ValueError, match="chunks_sha256"):
        assert_live_index_profile("raw-v1")
