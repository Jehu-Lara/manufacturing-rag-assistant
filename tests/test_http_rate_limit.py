from __future__ import annotations

from src.adapters.primary.http.rate_limit import RateLimiter


def test_distinct_keys_get_independent_budgets():
    limiter = RateLimiter(max_requests=2)

    assert limiter.allow("session:a") is True
    assert limiter.allow("session:a") is True
    assert limiter.allow("session:a") is False
    assert limiter.allow("session:b") is True


def test_window_expiry_lets_a_key_through_again():
    limiter = RateLimiter(max_requests=1, window_seconds=0.0)

    assert limiter.allow("session:a") is True
    assert limiter.allow("session:a") is True


def test_prune_drops_keys_whose_window_fully_expired():
    """The limiter is per-session now, so keys are unbounded in principle; a
    map that only ever grew would be a slow leak in a long-lived container."""
    limiter = RateLimiter(max_requests=5, window_seconds=0.0)
    limiter.allow("session:a")
    limiter.allow("session:b")

    limiter.prune()

    assert limiter._hits == {}


def test_prune_keeps_keys_still_inside_their_window():
    limiter = RateLimiter(max_requests=5, window_seconds=600.0)
    limiter.allow("session:a")

    limiter.prune()

    assert "session:a" in limiter._hits


def test_a_rejected_key_still_counts_against_itself():
    limiter = RateLimiter(max_requests=1, window_seconds=600.0)
    limiter.allow("session:a")

    assert limiter.allow("session:a") is False
    assert limiter.allow("session:a") is False
    assert len(limiter._hits["session:a"]) == 1


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_allow_evicts_stale_sessions_without_an_explicit_prune_call():
    """The reason `prune` exists is per-session keys: one entry per browser
    session ever seen, forever, on a public demo. A `prune` that only tests
    call would leave exactly the leak it was written to close."""
    clock = _FakeClock()
    limiter = RateLimiter(max_requests=5, window_seconds=60.0, clock=clock)

    for index in range(50):
        limiter.allow(f"session:{index}")
    assert len(limiter._hits) == 50

    clock.advance(61.0)
    limiter.allow("session:fresh")

    assert list(limiter._hits) == ["session:fresh"]


def test_allow_sweeps_at_most_once_per_window(monkeypatch):
    """A sweep on every request would be O(sessions) per call. Bounding it to
    one pass per window is what makes it cheap enough for the request path."""
    clock = _FakeClock()
    sweeps: list[float] = []
    original_prune = RateLimiter.prune

    def counting_prune(self: RateLimiter) -> None:
        sweeps.append(clock.now)
        original_prune(self)

    monkeypatch.setattr(RateLimiter, "prune", counting_prune)
    limiter = RateLimiter(max_requests=100, window_seconds=60.0, clock=clock)

    for _ in range(20):
        clock.advance(5.0)
        limiter.allow("session:a")

    assert sweeps == [1060.0]


def test_a_session_still_inside_its_window_survives_a_sweep():
    """The sweep must evict on staleness, not on being present when it runs."""
    clock = _FakeClock()
    limiter = RateLimiter(max_requests=5, window_seconds=60.0, clock=clock)
    limiter.allow("session:old")
    clock.advance(40.0)
    limiter.allow("session:recent")

    clock.advance(25.0)
    limiter.allow("session:new")

    assert "session:old" not in limiter._hits
    assert "session:recent" in limiter._hits
    assert "session:new" in limiter._hits
