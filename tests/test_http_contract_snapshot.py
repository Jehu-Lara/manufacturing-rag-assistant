"""Phase 0 freeze: pins GET /health and POST /query's exact JSON contract before
the src/ migration starts, so every later phase can prove it never changed the
wire shape, only the code location behind it. Run as a script
(`python -m tests.test_http_contract_snapshot`) to regenerate the stored
snapshot; run under pytest to verify the live app still reproduces it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.config import Settings
from api.main import app
from retrieval.hybrid import RetrievalResult

SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshots" / "http_contract_phase0.json"
REQUEST_ID_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"


def _mask_request_id(response_json: dict) -> dict:
    if isinstance(response_json, dict) and "request_id" in response_json:
        response_json = {**response_json, "request_id": REQUEST_ID_PLACEHOLDER}
    return response_json


def _settings(threshold: float, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        groq_api_key="fake-key",
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=threshold,
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


def _capture_health_normal() -> dict:
    with patch("retrieval.vector_store.get_collection", return_value=MagicMock()):
        with TestClient(app) as client:
            response = client.get("/health")
    return {
        "case": "health_normal",
        "request": {"method": "GET", "path": "/health"},
        "response_status": response.status_code,
        "response_json": response.json(),
    }


def _capture_health_degraded() -> dict:
    with patch("retrieval.vector_store.get_collection", side_effect=RuntimeError("no such collection")):
        with TestClient(app) as client:
            response = client.get("/health")
    return {
        "case": "health_degraded",
        "request": {"method": "GET", "path": "/health"},
        "response_status": response.status_code,
        "response_json": response.json(),
    }


def _capture_query_confident() -> dict:
    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}
    with patch("api.main.load_settings", return_value=_settings(threshold=0.3)):
        with patch("retrieval.hybrid.retrieve", return_value=[_retrieval_result("chunk-abc", 0.9)]):
            with patch(
                "api.llm_client.generate_structured",
                return_value={
                    "answer": "The QC unit is responsible for approving or rejecting components.",
                    "citations": [{"chunk_id": "chunk-abc"}],
                    "refused": False,
                },
            ):
                with TestClient(app) as client:
                    response = client.post("/query", json=request_body)
    return {
        "case": "query_confident",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": _mask_request_id(response.json()),
    }


def _capture_query_refused() -> dict:
    request_body = {"question": "What's the weather today?", "language": "en"}
    with patch("api.main.load_settings", return_value=_settings(threshold=0.7)):
        with patch("retrieval.hybrid.retrieve", return_value=[_retrieval_result("chunk-low", 0.1)]):
            with patch("api.llm_client.generate_structured") as mock_generate:
                with TestClient(app) as client:
                    response = client.post("/query", json=request_body)
    mock_generate.assert_not_called()
    return {
        "case": "query_refused",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": _mask_request_id(response.json()),
    }


def _capture_query_generation_error() -> dict:
    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}
    with patch("api.main.load_settings", return_value=_settings(threshold=0.3)):
        with patch("retrieval.hybrid.retrieve", return_value=[_retrieval_result("chunk-abc", 0.9)]):
            with patch("api.llm_client.generate_structured") as mock_generate:
                from api.llm_client import GenerationError

                mock_generate.side_effect = GenerationError("provider unavailable")
                with TestClient(app) as client:
                    response = client.post("/query", json=request_body)
    return {
        "case": "query_generation_error",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": _mask_request_id(response.json()),
    }


def _capture_query_unhandled_exception_500() -> dict:
    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}
    with patch("api.generation.answer_question", side_effect=RuntimeError("index not built")):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/query", json=request_body)
    return {
        "case": "query_unhandled_exception_500",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": response.json(),
    }


def _capture_all_cases() -> list[dict]:
    return [
        _capture_health_normal(),
        _capture_health_degraded(),
        _capture_query_confident(),
        _capture_query_refused(),
        _capture_query_generation_error(),
        _capture_query_unhandled_exception_500(),
    ]


def _load_snapshot() -> list[dict]:
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_live_app_matches_phase0_snapshot() -> None:
    live_cases = _capture_all_cases()
    stored_cases = _load_snapshot()
    assert live_cases == stored_cases


def _new_app_health_normal() -> dict:
    from src.adapters.primary.http.deps import get_vector_store
    from src.main import app
    from tests.fakes import InMemoryVectorStore

    app.dependency_overrides[get_vector_store] = lambda: InMemoryVectorStore(ready=True)
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    return {
        "case": "health_normal",
        "request": {"method": "GET", "path": "/health"},
        "response_status": response.status_code,
        "response_json": response.json(),
    }


def _new_app_health_degraded() -> dict:
    from src.adapters.primary.http.deps import get_vector_store
    from src.main import app
    from tests.fakes import InMemoryVectorStore

    app.dependency_overrides[get_vector_store] = lambda: InMemoryVectorStore(ready=False)
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    return {
        "case": "health_degraded",
        "request": {"method": "GET", "path": "/health"},
        "response_status": response.status_code,
        "response_json": response.json(),
    }


def _new_app_query_confident() -> dict:
    from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter, get_settings
    from src.adapters.primary.http.rate_limit import RateLimiter
    from src.features.query.use_cases import QueryUseCase
    from src.main import app
    from tests.fakes import InMemoryLLMClient, InMemoryRetriever

    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}
    settings = _settings(threshold=0.3)
    retriever = InMemoryRetriever([_retrieval_result("chunk-abc", 0.9)])
    llm = InMemoryLLMClient(
        response={
            "answer": "The QC unit is responsible for approving or rejecting components.",
            "citations": [{"chunk_id": "chunk-abc"}],
            "refused": False,
        }
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_query_use_case] = lambda: QueryUseCase(retriever, llm, settings)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=1000)
    try:
        with TestClient(app) as client:
            response = client.post("/query", json=request_body)
    finally:
        app.dependency_overrides.clear()
    return {
        "case": "query_confident",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": _mask_request_id(response.json()),
    }


def _new_app_query_refused() -> dict:
    from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter, get_settings
    from src.adapters.primary.http.rate_limit import RateLimiter
    from src.features.query.use_cases import QueryUseCase
    from src.main import app
    from tests.fakes import InMemoryLLMClient, InMemoryRetriever

    request_body = {"question": "What's the weather today?", "language": "en"}
    settings = _settings(threshold=0.7)
    retriever = InMemoryRetriever([_retrieval_result("chunk-low", 0.1)])
    llm = InMemoryLLMClient()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_query_use_case] = lambda: QueryUseCase(retriever, llm, settings)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=1000)
    try:
        with TestClient(app) as client:
            response = client.post("/query", json=request_body)
    finally:
        app.dependency_overrides.clear()
    return {
        "case": "query_refused",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": _mask_request_id(response.json()),
    }


def _new_app_query_generation_error() -> dict:
    from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter, get_settings
    from src.adapters.primary.http.rate_limit import RateLimiter
    from src.core.errors import GenerationError
    from src.features.query.use_cases import QueryUseCase
    from src.main import app
    from tests.fakes import InMemoryLLMClient, InMemoryRetriever

    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}
    settings = _settings(threshold=0.3)
    retriever = InMemoryRetriever([_retrieval_result("chunk-abc", 0.9)])
    llm = InMemoryLLMClient(error=GenerationError("provider unavailable"))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_query_use_case] = lambda: QueryUseCase(retriever, llm, settings)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=1000)
    try:
        with TestClient(app) as client:
            response = client.post("/query", json=request_body)
    finally:
        app.dependency_overrides.clear()
    return {
        "case": "query_generation_error",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": _mask_request_id(response.json()),
    }


def _new_app_query_unhandled_exception_500() -> dict:
    from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter
    from src.adapters.primary.http.rate_limit import RateLimiter
    from src.main import app

    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}

    class _RaisingUseCase:
        async def answer_question(self, question: str, language: str):
            raise RuntimeError("index not built")

    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_requests=1000)
    app.dependency_overrides[get_query_use_case] = lambda: _RaisingUseCase()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/query", json=request_body)
    finally:
        app.dependency_overrides.clear()
    return {
        "case": "query_unhandled_exception_500",
        "request": {"method": "POST", "path": "/query", "json": request_body},
        "response_status": response.status_code,
        "response_json": response.json(),
    }


def test_new_app_matches_phase0_snapshot() -> None:
    """Phase 2 gate: proves src.main:app's POST /query and GET /health JSON
    shapes are byte-identical to the Phase 0 snapshot captured from the old
    api.main:app — only the code location behind them changed."""
    new_app_cases = [
        _new_app_health_normal(),
        _new_app_health_degraded(),
        _new_app_query_confident(),
        _new_app_query_refused(),
        _new_app_query_generation_error(),
        _new_app_query_unhandled_exception_500(),
    ]
    stored_cases = _load_snapshot()
    assert new_app_cases == stored_cases


if __name__ == "__main__":
    cases = _capture_all_cases()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {len(cases)} cases to {SNAPSHOT_PATH}")
