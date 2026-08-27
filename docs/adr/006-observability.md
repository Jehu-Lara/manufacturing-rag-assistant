# ADR-006: Observability — OTel spans + JSON logs with trace_id

## Context

Before this refactor, JSON logging (`api/logging_setup.py`'s `JsonFormatter`)
had no distributed tracing — no way to correlate a single `/query` request's
retrieval, LLM-call, and completion log lines beyond the manually-generated
`request_id` field threaded through each log call by hand.

## Decision

`src/core/logging.py`'s `JsonFormatter` reads the active OpenTelemetry
span's trace id (via `opentelemetry.trace.get_current_span()`) into a
`trace_id` log field whenever a valid span context exists — a no-op
(verified directly: `get_span_context().is_valid` is `False` with no
tracer configured) until spans are actually created. `src/core/telemetry.py`
provides `configure_tracing(app)` as the FastAPI instrumentation entry
point; `src/adapters/primary/http/app.py`'s `create_app()` calls it.

## Consequences

- Log lines automatically carry `trace_id` once a request is inside a span,
  without every call site needing to pass one explicitly — `request_id`
  (business-level, generated once per query) and `trace_id`
  (infrastructure-level, per span) are complementary, not duplicates.
- `HybridRetriever.retrieve()` (sentence-transformers encode + Chroma query
  + BM25 search) stays synchronous inside `QueryUseCase`'s `async def
  answer_question`, wrapped in `asyncio.to_thread` — documented here
  explicitly so it isn't mistaken for an oversight: none of the pinned
  library versions (sentence-transformers, chromadb) expose an async API,
  and the single-process/single-worker deploy shape (ADR-002, ADR-005)
  already means one request is served at a time per worker regardless.
- An OTel exporter target (`OTEL_EXPORTER_OTLP_ENDPOINT`) needs a real
  collector in production; unset, tracing is a local no-op — logs still
  work exactly as before, just without a populated `trace_id` field. This
  is an intentionally simple setup, appropriate for a portfolio-scale app,
  not a production observability stack.
