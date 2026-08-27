# ADR-002: In-memory, per-process rate limiting

## Context

`POST /query` needs basic abuse protection against a public demo endpoint.
The currently verified deploy image, targeting a single Oracle Cloud VM,
runs one container with one uvicorn worker — there is no shared store today
to keep a rate limit in sync across replicas even if there were more than one.

## Decision

Keep the existing in-memory, per-process, sliding-window `RateLimiter`
(`src/adapters/primary/http/rate_limit.py`) — a `deque` per client key,
evicting hits older than the window on each check. No Redis, no external
store.

## Consequences

- Correct and sufficient for exactly one process. Would under-count or
  over-allow if this were ever horizontally scaled to multiple
  workers/replicas — explicitly out of scope until a real scaling need
  exists (see ADR-005's deferred deploy-shape decision).
- **Caveat identified at review, documentation-only, no code change:**
  behind the current single-container nginx proxy, `src/web` (Streamlit)
  calls the API over loopback (`127.0.0.1`). The limiter keys on
  `http_request.client.host`, so every public Streamlit visitor arrives at
  the API as the same loopback address — the limiter effectively caps the
  UI process as a whole, not individual public users. This is a real
  limitation of the current deploy shape, not something this refactor
  fixes; a per-session limiter inside `src/web` itself is legitimate future
  work, tracked separately, not mixed into this structural refactor.
