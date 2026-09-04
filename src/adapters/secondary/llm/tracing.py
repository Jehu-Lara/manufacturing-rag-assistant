from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmTraceEvent:
    """Immutable, content-free record of one physical step of a generation
    call. Carries provider/model, token counts, finish reason and error
    shape — never the prompt, the question, the answer, an API key, or a raw
    exception string. `str(exc)` in particular is deliberately absent: a
    provider's error body can echo prompt fragments back, so only the exception
    TYPE and HTTP status are captured (see `_provider_error_trace_fields`).

    Consumed by an injected `trace_hook`. Two consumers exist: the Phase 3C
    generation runner, which buckets events for per-call accounting, and
    `log_llm_trace` below, which production wires in at the composition root
    (`src.adapters.primary.http.app.lifespan`) so these events reach the JSON
    log stream. This dataclass's field list is therefore a real disclosure
    boundary, not just an internal record — adding a field to it adds that
    field to production's stdout."""

    event: str
    provider: str
    phase: str
    model: Optional[str] = None
    attempt: Optional[int] = None
    wait_seconds: Optional[float] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    system_fingerprint: Optional[str] = None
    schema_mode: Optional[str] = None
    error_type: Optional[str] = None
    status_code: Optional[int] = None


TraceHook = Callable[[LlmTraceEvent], None]


def log_llm_trace(event: LlmTraceEvent) -> None:
    """Production trace sink. Safe to wire in by construction, not by review:
    LlmTraceEvent's fields are the only thing it can emit, and that dataclass
    holds provider/model/token/latency/error-shape values only — never a
    prompt, a question, an answer, an API key, or str(exc). Until this existed,
    the per-physical-call picture (retries, 429s, schema fallbacks, provider
    failovers) was visible only to the Phase 3C eval runner, so production had
    no way to see a call that was retried four times before succeeding."""
    fields = {
        f"llm_{key}": value
        for key, value in asdict(event).items()
        if key != "event" and value is not None
    }
    logger.info("llm trace", extra={"event": "llm_trace", "llm_event": event.event, **fields})


# Groq free-tier chat model with tool/JSON-schema support. The original pick
# (llama-3.3-70b-versatile) was retired by Groq and started 404ing on every
# call — confirmed live against GET https://api.groq.com/openai/v1/models on
# 2026-08-25, and against a real generate_structured() call with the JSON
# schema response_format, both succeeding. Model availability on a free-
# tier provider can change without notice; re-verify this the same way if a


def _provider_error_fields(provider: str, exc: Exception) -> dict[str, object]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) if response is not None else None
    fields: dict[str, object] = {"provider": provider, "error_type": type(exc).__name__}
    if isinstance(status_code, int):
        fields["status_code"] = status_code
    return fields


def _provider_error_trace_fields(exc: Exception) -> dict[str, Any]:
    """Only the exception TYPE and HTTP status — never str(exc), which can
    echo prompt fragments or a key back from a provider error body."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) if response is not None else None
    fields: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(status_code, int):
        fields["status_code"] = status_code
    return fields


def _usage_trace_fields(response: Any, model: str, schema_mode: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    choices = getattr(response, "choices", None) or []
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    return {
        "model": model,
        "schema_mode": schema_mode,
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "system_fingerprint": (
            getattr(response, "system_fingerprint", None)
            if isinstance(getattr(response, "system_fingerprint", None), str)
            else None
        ),
    }
