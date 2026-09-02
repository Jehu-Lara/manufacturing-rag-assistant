from __future__ import annotations

import httpx

from src.web import client


def _mock_transport(status_code: int, json_body: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_body)

    return httpx.MockTransport(handler)


def test_query_returns_200_answer_response():
    body = {
        "answer": "The QC unit is responsible for X.",
        "citations": [],
        "refused": False,
        "status": "ok",
        "confidence": 0.9,
        "threshold": 0.5999,
        "language": "en",
        "request_id": "req-1",
    }
    transport = _mock_transport(200, body)

    response = client.query("What is the QC unit responsible for?", "en", transport=transport)

    assert response.status_code == 200
    assert response.json() == body


def test_query_returns_refused_response():
    body = {
        "answer": "I don't have enough information...",
        "citations": [],
        "refused": True,
        "status": "ok",
        "confidence": 0.1,
        "threshold": 0.5999,
        "language": "en",
        "request_id": "req-2",
    }
    transport = _mock_transport(200, body)

    response = client.query("Unanswerable question", "en", transport=transport)

    assert response.json()["refused"] is True


def test_query_returns_429_rate_limited():
    transport = _mock_transport(429, {"detail": "Rate limit exceeded. Try again shortly."})

    response = client.query("Any question", "en", transport=transport)

    assert response.status_code == 429


def test_ready_returns_true_on_200():
    transport = _mock_transport(200, {"status": "ready"})
    assert client.ready(transport=transport) is True


def test_ready_returns_false_on_503():
    transport = _mock_transport(503, {"status": "not_ready"})
    assert client.ready(transport=transport) is False


def test_ready_returns_false_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    assert client.ready(transport=transport) is False


def test_health_returns_response_on_200():
    transport = _mock_transport(
        200, {"status": "ok", "embedding_model": "x", "llm_provider_primary": "groq", "index_loaded": True}
    )
    response = client.health(transport=transport)
    assert response is not None
    assert response.status_code == 200


def test_health_returns_none_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    assert client.health(transport=transport) is None


def test_query_sends_api_key_header_when_configured(monkeypatch):
    monkeypatch.setattr(client, "API_KEY", "secret-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client.query("q", "en", transport=transport)

    assert captured["headers"].get("x-api-key") == "secret-key"


def test_query_omits_api_key_header_when_not_configured(monkeypatch):
    monkeypatch.setattr(client, "API_KEY", None)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client.query("q", "en", transport=transport)

    assert "x-api-key" not in captured["headers"]


def test_query_sends_client_session_header_when_given(monkeypatch):
    monkeypatch.setattr(client, "API_KEY", None)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    client.query("q", "en", session_id="11111111-1111-4111-8111-111111111111", transport=httpx.MockTransport(handler))

    assert captured["headers"].get("x-client-session") == "11111111-1111-4111-8111-111111111111"


def test_query_omits_client_session_header_when_absent(monkeypatch):
    monkeypatch.setattr(client, "API_KEY", None)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    client.query("q", "en", transport=httpx.MockTransport(handler))

    assert "x-client-session" not in captured["headers"]
