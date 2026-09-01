from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.adapters.primary.http import app as app_module
from tests.conftest import REQUIRES_BUILT_INDEX_REASON, built_retrieval_index_present

pytestmark = pytest.mark.skipif(
    not built_retrieval_index_present(), reason=REQUIRES_BUILT_INDEX_REASON
)


def _raise(message: str):
    def _boom(*args: object, **kwargs: object):
        raise RuntimeError(message)

    return _boom


def test_lifespan_aborts_when_manifest_verify_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module.index_manifest, "verify", _raise("manifest drift"))

    with pytest.raises(RuntimeError, match="manifest drift"):
        with TestClient(app_module.create_app()):
            pass


def test_lifespan_aborts_when_collection_validation_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        app_module.ChromaVectorStore, "validate_collection", _raise("wrong profile")
    )

    with pytest.raises(RuntimeError, match="wrong profile"):
        with TestClient(app_module.create_app()):
            pass


def test_lifespan_aborts_when_bm25_validation_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module.Bm25LexicalIndex, "validate", _raise("bm25 mismatch"))

    with pytest.raises(RuntimeError, match="bm25 mismatch"):
        with TestClient(app_module.create_app()):
            pass


def test_lifespan_succeeds_against_the_built_index():
    with TestClient(app_module.create_app()) as client:
        assert client.get("/health").status_code == 200
