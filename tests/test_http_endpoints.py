from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter, get_settings, get_vector_store
from src.adapters.primary.http.rate_limit import RateLimiter
from src.core.config import Settings, load_settings
from src.domain.models import RetrievalResult
from src.features.query.use_cases import QueryUseCase
from src.main import app
from tests.fakes import InMemoryLLMClient, InMemoryRetriever, InMemoryVectorStore


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _settings(**overrides) -> Settings:
    base = dict(
        groq_api_key="fake-key",
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=0.3,
        log_level="INFO",
    )
    base.update(overrides)
    return Settings(**base)


def _retrieval_result(chunk_id: str, semantic_score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=1.0,
        semantic_rank=1,
        semantic_score=semantic_score,
        bm25_rank=1,
        bm25_score=1.0,
        metadata={
            "document_id": "doc-real",
            "document_title": "Real Retrieved Title",
            "section_heading": "Real Section",
            "revision": "Rev Z",
            "chunk_id": chunk_id,
            "chunk_text": "some real chunk text",
        },
    )


def test_health_returns_200_with_index_loaded_true_when_ping_succeeds(client):
    app.dependency_overrides[get_vector_store] = lambda: InMemoryVectorStore(ready=True)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True


def test_health_returns_200_with_index_loaded_false_when_ping_fails(client):
    app.dependency_overrides[get_vector_store] = lambda: InMemoryVectorStore(ready=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["index_loaded"] is False


def test_ready_returns_200_when_vector_store_ready(client):
    app.dependency_overrides[get_vector_store] = lambda: InMemoryVectorStore(ready=True)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_vector_store_not_ready(client):
    app.dependency_overrides[get_vector_store] = lambda: InMemoryVectorStore(ready=False)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_query_answerable_returns_200(client):
    settings = _settings()
    retriever = InMemoryRetriever([_retrieval_result("chunk-abc", 0.9)])
    llm = InMemoryLLMClient(
        response={
            "answer": "The QC unit is responsible for X.",
            "citations": [{"chunk_id": "chunk-abc"}],
            "refused": False,
        }
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_query_use_case] = lambda: QueryUseCase(retriever, llm, settings)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=100)

    response = client.post("/query", json={"question": "What is the QC unit responsible for?", "language": "en"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["citations"][0]["document_id"] == "doc-real"


def test_query_missing_question_returns_422(client):
    response = client.post("/query", json={"language": "en"})
    assert response.status_code == 422


def test_query_invalid_language_returns_422(client):
    response = client.post("/query", json={"question": "What is the SOP?", "language": "fr"})
    assert response.status_code == 422


def test_query_exceeding_rate_limit_returns_429(client):
    settings = _settings()
    retriever = InMemoryRetriever([_retrieval_result("chunk-abc", 0.9)])
    llm = InMemoryLLMClient(response={"answer": "ok", "citations": [{"chunk_id": "chunk-abc"}], "refused": False})
    rate_limiter = RateLimiter(max_requests=2)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_query_use_case] = lambda: QueryUseCase(retriever, llm, settings)
    app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter

    payload = {"question": "What is the QC unit responsible for?", "language": "en"}
    first = client.post("/query", json=payload)
    second = client.post("/query", json=payload)
    third = client.post("/query", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429


def test_query_with_configured_api_key_rejects_missing_header_with_401(client):
    settings = _settings(api_key="secret-key")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=100)

    response = client.post("/query", json={"question": "What is the QC unit responsible for?", "language": "en"})

    assert response.status_code == 401


def test_query_with_configured_api_key_rejects_wrong_header_with_401(client):
    settings = _settings(api_key="secret-key")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=100)

    response = client.post(
        "/query",
        json={"question": "What is the QC unit responsible for?", "language": "en"},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_query_with_configured_api_key_accepts_matching_header(client):
    settings = _settings(api_key="secret-key")
    retriever = InMemoryRetriever([_retrieval_result("chunk-abc", 0.9)])
    llm = InMemoryLLMClient(response={"answer": "ok", "citations": [{"chunk_id": "chunk-abc"}], "refused": False})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_query_use_case] = lambda: QueryUseCase(retriever, llm, settings)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=100)

    response = client.post(
        "/query",
        json={"question": "What is the QC unit responsible for?", "language": "en"},
        headers={"X-API-Key": "secret-key"},
    )

    assert response.status_code == 200


def test_query_unhandled_exception_returns_500_with_generic_body():
    class _RaisingUseCase:
        async def answer_question(self, question, language):
            raise RuntimeError("index not built")

    no_key_settings = load_settings().model_copy(update={"api_key": None})
    app.dependency_overrides[get_settings] = lambda: no_key_settings
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=100)
    app.dependency_overrides[get_query_use_case] = lambda: _RaisingUseCase()
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post(
                "/query", json={"question": "What is the QC unit responsible for?", "language": "en"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"status": "error", "detail": "internal server error"}


def test_auth_uses_constant_time_comparison_not_plain_inequality():
    """secrets.compare_digest is used (Resolved Decision #3), not `!=` — this
    doesn't change observable behavior, so assert it by inspecting the
    router source rather than timing, which would be flaky."""
    import inspect

    from src.features.query import router as router_module

    source = inspect.getsource(router_module)
    assert "secrets.compare_digest" in source
    assert "x_api_key != settings.api_key" not in source


def test_framework_documentation_routes_are_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_chromadb_server_api_is_not_mounted(client):
    response = client.post(
        "/api/v2/tenants/default_tenant/databases/default_database/collections",
        json={"name": "untrusted", "get_or_create": True},
    )

    assert response.status_code == 404


def test_fastapi_does_not_duplicate_the_nginx_edge_security_headers(client):
    """Defensive headers are the public nginx edge's job (single authority);
    FastAPI must not re-add them, or /health and /ready would carry a
    duplicated X-Frame-Options / CSP through the proxy."""
    response = client.get("/health")
    assert "X-Frame-Options" not in response.headers
    assert "Content-Security-Policy" not in response.headers
