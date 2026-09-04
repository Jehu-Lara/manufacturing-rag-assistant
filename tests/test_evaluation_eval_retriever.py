from __future__ import annotations

import pytest

from src.adapters.secondary.lexical.bm25_lexical_index import BM25_SCHEMA_VERSION, LEXICAL_PROFILE
from src.features.evaluation import _eval_retriever
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
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
        lexical_profile=LEXICAL_PROFILE,
        bm25_schema_version=BM25_SCHEMA_VERSION,
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


class _StubSettings:
    chroma_path = "chroma"
    bm25_path = "bm25"


class _StubVectorStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.validated: object = None

    def validate_collection(self, *, expected_profile: str, expected_count: int) -> None:
        self.validated = (expected_profile, expected_count)


class _StubBm25:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.validated: object = None

    def validate(self, expected_chunk_ids: list[str], **kwargs: object) -> None:
        self.validated = (expected_chunk_ids, kwargs)


def _patch_build_happy_path(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}
    monkeypatch.setattr(_eval_retriever, "load_settings", lambda: _StubSettings())
    monkeypatch.setattr(_eval_retriever, "SentenceTransformersEmbedder", lambda *a, **k: object())
    monkeypatch.setattr(_eval_retriever, "ChromaVectorStore", _StubVectorStore)
    monkeypatch.setattr(_eval_retriever, "Bm25LexicalIndex", _StubBm25)
    monkeypatch.setattr(_eval_retriever, "load_chunks", lambda: [])
    monkeypatch.setattr(
        _eval_retriever, "HybridRetriever", lambda vs, li, expansion_mode="off": ("retriever", expansion_mode)
    )
    monkeypatch.setattr(
        _eval_retriever.index_manifest, "resolve_index_profile", lambda *_: "contextual-v1"
    )
    monkeypatch.setattr(
        _eval_retriever.index_manifest,
        "verify",
        lambda **kw: (calls.__setitem__("verify_kw", kw), _fake_manifest("contextual-v1"))[1],
    )
    return calls


def test_build_retriever_runs_physical_coherence_check(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_build_happy_path(monkeypatch)
    result = build_retriever("semantic")
    assert result == ("retriever", "semantic")
    assert calls["verify_kw"] == {"expected_profile": "contextual-v1"}


def test_build_retriever_propagates_manifest_verify_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build_happy_path(monkeypatch)

    def _boom(**kw: object) -> object:
        raise ValueError("chunk_count stored 700, computed 42")

    monkeypatch.setattr(_eval_retriever.index_manifest, "verify", _boom)
    with pytest.raises(ValueError, match="chunk_count"):
        build_retriever()


def test_build_retriever_propagates_collection_validation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build_happy_path(monkeypatch)

    class _BadStore(_StubVectorStore):
        def validate_collection(self, *, expected_profile: str, expected_count: int) -> None:
            raise RuntimeError("live collection index_profile is 'raw-v1', expected 'contextual-v1'")

    monkeypatch.setattr(_eval_retriever, "ChromaVectorStore", _BadStore)
    with pytest.raises(RuntimeError, match="index_profile"):
        build_retriever()


def test_build_retriever_propagates_bm25_validation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build_happy_path(monkeypatch)

    class _BadBm25(_StubBm25):
        def validate(self, expected_chunk_ids: list[str], **kwargs: object) -> None:
            raise RuntimeError("BM25 chunk ids do not match the indexed chunks")

    monkeypatch.setattr(_eval_retriever, "Bm25LexicalIndex", _BadBm25)
    with pytest.raises(RuntimeError, match="BM25 chunk ids"):
        build_retriever()


def test_build_retriever_can_skip_the_coherence_check(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build_happy_path(monkeypatch)

    def _should_not_run(**kw: object) -> object:
        raise AssertionError("verify must not be called when coherence check is disabled")

    monkeypatch.setattr(_eval_retriever.index_manifest, "verify", _should_not_run)
    assert build_retriever(verify_physical_coherence=False) == ("retriever", "off")
