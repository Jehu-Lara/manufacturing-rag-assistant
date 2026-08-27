from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """In-memory sliding-window limiter keyed by client identity. Scoped to a
    single process by design: this deploys as one container with one uvicorn
    worker, so there is no shared store to keep in sync across replicas (see
    ADR-002). Behind the current single-container nginx proxy, Streamlit
    calls the API over loopback, so this limiter sees every public Streamlit
    visitor as the same client IP — it rate-limits the UI process as a
    whole, not individual public users; documented as a known limitation of
    the current deploy shape, not fixed here."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self._window_seconds:
            hits.popleft()
        if len(hits) >= self._max_requests:
            return False
        hits.append(now)
        return True
