# ADR-006: Observability — OTel spans + JSON logs with trace_id

## Context

Before this refactor, JSON logging (`api/logging_setup.py`'s `JsonFormatter`)
had no distributed tracing — no way to correlate a single `/query` request's
retrieval, LLM-call, and completion log lines beyond the manually-generated
`request_id` field threaded through each log call by hand.

## Decision

`src/core/logging.py`'s `JsonFormatter` reads the active OpenTelemetry
span's trace id (via `opentelemetry.trace.get_current_span()`) into a
`trace_id` log field whenever a valid span context exists. `src/core/telemetry.py`'s
`configure_tracing(app)` sets up a real `TracerProvider` (called once from
`create_app()`; idempotent — a second call is a documented no-op, since
OTel itself silently ignores a second `set_tracer_provider` in the same
process) and, only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, adds an OTLP
exporter — unset, spans are still created (so `trace_id` still populates)
but never leave the process.

Three named spans are created via `get_tracer().start_as_current_span(...)`:
`retrieval.hybrid.query` (`HybridRetriever.retrieve`), `embedder.compute`
(`SentenceTransformersEmbedder.embed_texts`), and `llm.generate`
(`GroqOpenAiLlmClient.generate_structured`), all nested under one outer
`query.answer_question` span opened in `QueryUseCase.answer_question` —
verified this nesting survives the `asyncio.to_thread` boundary the
retriever call crosses, since `asyncio.to_thread` copies the current
`contextvars.Context` (which is how OTel tracks the active span) to its
worker thread, per the stdlib's own documented behavior. Tests
(`tests/test_core_telemetry.py`) verify each span name via a mocked
tracer, not a real exporter — mutating the process-global
`TracerProvider` per-test isn't reliable (same silent-second-call
behavior noted above).

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
