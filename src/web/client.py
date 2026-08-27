from __future__ import annotations

import os
from typing import Optional

import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY") or None
# Carried forward unchanged from the pre-move ui/streamlit_app.py — this
# already existed and was never the gap a reviewer flagged (verified against
# the actual file before this move); not a new fix.
REQUEST_TIMEOUT_SECONDS = 60.0

# A separate, short timeout for /ready — main() calls this once per Streamlit
# rerun (every widget interaction) to gate the Ask button, so a slow/hung
# backend must not stall every single page interaction for the full 60s
# query timeout above.
READY_CHECK_TIMEOUT_SECONDS = 5.0


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def query(question: str, language: str, transport: Optional[httpx.BaseTransport] = None) -> httpx.Response:
    """`transport` is test-only (httpx.MockTransport) — omitted, this uses
    httpx's real default transport, identical to calling httpx.post directly."""
    with httpx.Client(transport=transport) as client:
        return client.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "language": language},
            headers=_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )


def health(transport: Optional[httpx.BaseTransport] = None) -> Optional[httpx.Response]:
    try:
        with httpx.Client(transport=transport) as client:
            return client.get(f"{API_BASE_URL}/health", timeout=READY_CHECK_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return None


def ready(transport: Optional[httpx.BaseTransport] = None) -> bool:
    """Additive GET /ready (Phase 2) — used to gate the Ask button instead of
    submitting a query against a backend whose index never loaded."""
    try:
        with httpx.Client(transport=transport) as client:
            response = client.get(f"{API_BASE_URL}/ready", timeout=READY_CHECK_TIMEOUT_SECONDS)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
