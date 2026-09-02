from __future__ import annotations

import time
from collections import deque
from typing import Callable


class RateLimiter:
    """In-memory sliding-window limiter keyed by client identity. Scoped to a
    single process by design: this deploys as one container with one uvicorn
    worker, so there is no shared store to keep in sync across replicas (see
    ADR-002).

    `key` is supplied by the caller. Behind the single-container nginx proxy
    every public visitor reaches the API over loopback from the same Streamlit
    process, so an IP-derived key would put all of them in one bucket; the
    router therefore prefers a per-browser-session key (see
    `src.features.query.router._rate_limit_key`)."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`clock` is a seam, not a feature: this is a time-dependent policy,
        and the only alternatives for testing eviction are sleeping through a
        real window or monkeypatching `time.monotonic` process-wide, which
        would also move the clock under pytest and asyncio."""
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._last_prune = clock()

    def allow(self, key: str) -> bool:
        """Explicit get/reinsert rather than a defaultdict: with per-session
        keys, every distinct key a caller ever presents would otherwise create
        a permanent empty deque, so the map would grow without bound on
        rejected or one-shot sessions. A key whose window has fully expired is
        dropped instead of being written back.

        That alone only bounds each individual key's deque — the map itself
        still grows one entry per session id ever seen, and session ids are
        minted per browser session, so a public demo accumulates them. Hence
        the sweep below: at most one full pass per window, on the request path,
        which is cheap enough for a single-process limiter and needs no
        background task to go wrong on its own."""
        now = self._clock()
        if now - self._last_prune >= self._window_seconds:
            self.prune()
        hits = self._hits.get(key, deque())
        while hits and now - hits[0] > self._window_seconds:
            hits.popleft()
        if len(hits) >= self._max_requests:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True

    def prune(self) -> None:
        """Drops keys whose window has fully expired. Called by `allow` at most
        once per window, and safe to call directly."""
        now = self._clock()
        self._last_prune = now
        for key in list(self._hits):
            hits = self._hits[key]
            while hits and now - hits[0] > self._window_seconds:
                hits.popleft()
            if not hits:
                del self._hits[key]
