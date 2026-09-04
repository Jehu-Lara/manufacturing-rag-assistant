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


def test_lifespan_wires_fail_fast_llm_backoff(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    real_client_cls = app_module.GroqOpenAiLlmClient

    class _Recording(real_client_cls):  # type: ignore[misc, valid-type]
        """A subclass, not a plain function: lifespan constructs the client
        through the `from_settings` classmethod, which a function stand-in
        would not carry."""

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module, "GroqOpenAiLlmClient", _Recording)

    with TestClient(app_module.create_app()):
        pass

    assert captured.get("rate_limit_backoff_seconds") == ()


def test_cors_preflight_allows_client_session_header(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://example.com")

    with TestClient(app_module.create_app()) as client:
        response = client.options(
            "/query",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Client-Session",
            },
        )

    assert response.status_code == 200
    assert "x-client-session" in response.headers["access-control-allow-headers"].lower()


def test_lifespan_closes_llm_client_on_partial_startup(monkeypatch: pytest.MonkeyPatch):
    closed: list[bool] = []
    real_cls = app_module.GroqOpenAiLlmClient

    class _Recording(real_cls):  # type: ignore[misc]
        async def aclose(self) -> None:
            closed.append(True)
            await super().aclose()

    monkeypatch.setattr(app_module, "GroqOpenAiLlmClient", _Recording)
    monkeypatch.setattr(app_module, "RateLimiter", _raise("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        with TestClient(app_module.create_app()):
            pass

    assert closed == [True]
