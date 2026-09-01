"""Phase 0 froze GET /health and POST /query's exact JSON contract by
capturing it from the old api.main:app (now retired — its replacement,
src.main:app, was proven byte-identical in Phase 2 and the old app no
longer exists after Phase 3's cutover). This file now only verifies the
current src.main:app still reproduces that frozen snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import REQUIRES_BUILT_INDEX_REASON, built_retrieval_index_present

pytestmark = pytest.mark.skipif(
    not built_retrieval_index_present(), reason=REQUIRES_BUILT_INDEX_REASON
)

SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshots" / "http_contract_phase0.json"
REQUEST_ID_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"

# Phase 3 (ADR-009) adds these to POST /query, additively. The Phase 0
# snapshot froze the legacy fields and their values, not the absence of
# backward-compatible extensions; the snapshot file is NOT rewritten.
PHASE3_ADDITIVE_KEYS = ("review_floor", "gate_band")


def _project_legacy(response_json: dict) -> dict:
    if not isinstance(response_json, dict):
        return response_json
    return {k: v for k, v in response_json.items() if k not in PHASE3_ADDITIVE_KEYS}


def _project_case_legacy(case: dict) -> dict:
    return {**case, "response_json": _project_legacy(case["response_json"])}


def _mask_request_id(response_json: dict) -> dict:
    if isinstance(response_json, dict) and "request_id" in response_json:
        response_json = {**response_json, "request_id": REQUEST_ID_PLACEHOLDER}
    return response_json


def _settings(threshold: float):
    from src.core.config import Settings

    return Settings(
        groq_api_key="fake-key",
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=threshold,
        log_level="INFO",
    )


def _retrieval_result(chunk_id: str, semantic_score: float):
    from src.domain.models import RetrievalResult

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


def _load_snapshot() -> list[dict]:
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _new_app_health_normal() -> dict:
    from fastapi.testclient import TestClient

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
    from fastapi.testclient import TestClient

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
    from fastapi.testclient import TestClient

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
    from fastapi.testclient import TestClient

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
    from fastapi.testclient import TestClient

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
    from fastapi.testclient import TestClient

    from src.adapters.primary.http.deps import get_query_use_case, get_rate_limiter, get_settings
    from src.adapters.primary.http.rate_limit import RateLimiter
    from src.core.config import load_settings
    from src.main import app

    request_body = {"question": "What is the QC unit responsible for?", "language": "en"}

    class _RaisingUseCase:
        async def answer_question(self, question: str, language: str):
            raise RuntimeError("index not built")

    no_key_settings = load_settings().model_copy(update={"api_key": None})
    app.dependency_overrides[get_settings] = lambda: no_key_settings
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


def _all_new_app_cases() -> list[dict]:
    return [
        _new_app_health_normal(),
        _new_app_health_degraded(),
        _new_app_query_confident(),
        _new_app_query_refused(),
        _new_app_query_generation_error(),
        _new_app_query_unhandled_exception_500(),
    ]


def test_new_app_matches_phase0_snapshot_on_legacy_fields() -> None:
    """Phase 2 gate, still enforced post-cutover: src.main:app's POST /query
    and GET /health legacy JSON fields+values stay byte-identical to the
    Phase 0 snapshot. Phase 3's additive keys are projected out first (ADR-009)
    and checked in test_phase3_additive_fields below; the snapshot file itself
    is never rewritten."""
    new_app_cases = [_project_case_legacy(case) for case in _all_new_app_cases()]
    stored_cases = _load_snapshot()
    assert new_app_cases == stored_cases


def test_phase3_additive_fields_present_on_query_and_absent_on_health() -> None:
    cases = {case["case"]: case["response_json"] for case in _all_new_app_cases()}

    for query_case in ("query_confident", "query_refused", "query_generation_error"):
        body = cases[query_case]
        assert set(PHASE3_ADDITIVE_KEYS) <= set(body), query_case
        assert body["gate_band"] in ("hard_refuse", "grounded_review", "confident")
        # default policy is binary -> review_floor is always null
        assert body["review_floor"] is None

    assert cases["query_confident"]["gate_band"] == "confident"
    assert cases["query_refused"]["gate_band"] == "hard_refuse"

    for health_case in ("health_normal", "health_degraded"):
        assert not (set(PHASE3_ADDITIVE_KEYS) & set(cases[health_case]))

    # decision_reason is internal only — it must never reach the HTTP body
    for body in cases.values():
        assert "decision_reason" not in body
