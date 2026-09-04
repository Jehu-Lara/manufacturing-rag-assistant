from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Optional, cast

import groq
import openai
from pydantic import SecretStr

from src.core.config import LlmProvider, Settings
from src.core.errors import GenerationError
from src.core.telemetry import get_tracer

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
# live run ever starts failing with "model ... does not exist" again.
GROQ_MODEL = "openai/gpt-oss-120b"

# response_format={"type": "json_schema", ...} is confirmed supported for
# this exact model per the task brief; no json_object fallback needed here.
OPENAI_MODEL = "gpt-4o-mini"

# Hard cap on generated tokens per call — bounds cost and runaway generation,
# not a target length. Set to 1024, not a tighter value, because the primary
# model (gpt-oss-120b) is a reasoning model: on Groq this budget also covers
# the hidden reasoning trace, which at the default reasoning effort routinely
# runs into the hundreds of tokens before any JSON is emitted. The eval-set
# gold answers serialize to <=160 tokens of response JSON, so 1024 leaves
# ample room for reasoning plus a thorough multi-citation answer while still
# capping the worst case. Re-tune against `generation_eval` (real LLM calls)
# if truncation (finish_reason "length") ever shows up in practice.
MAX_COMPLETION_TOKENS = 1024

# Fixed backoff schedule (seconds) used when a provider's rate-limit error
# carries no Retry-After value. Index 0 is the wait before the 1st retry,
# etc. len() also defines the max number of rate-limit retries per call.
RATE_LIMIT_BACKOFF_SECONDS = (15, 30, 60)

_RATE_LIMIT_ERROR_TYPES = (groq.RateLimitError, openai.RateLimitError)
_JSON_SCHEMA_RETRY_ERROR_TYPES = (groq.BadRequestError, groq.UnprocessableEntityError)


def _other_provider(provider: str) -> str:
    return "openai" if provider == "groq" else "groq"


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


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _is_unsupported_response_format_error(exc: Exception) -> bool:
    haystack = str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        haystack += " " + json.dumps(body)
    haystack = haystack.lower()
    return "response_format" in haystack or "json_schema" in haystack


def _try_parse_and_validate(
    raw_text: str, schema: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[Exception]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, exc
    try:
        _validate_against_schema(parsed, schema)
    except ValueError as exc:
        return None, exc
    return parsed, None


def _validate_against_schema(instance: object, schema: dict[str, Any]) -> None:
    """Hand-rolled structural check over the JSON Schema subset used by
    src.features.query.prompts.JSON_SCHEMA (object/array/string/boolean,
    properties, required, additionalProperties, items) — not a
    general-purpose validator. Raises ValueError with a human-readable
    reason on mismatch."""
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(instance, dict):
            raise ValueError(f"expected object, got {type(instance).__name__}")
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise ValueError(f"missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise ValueError(f"unexpected properties: {sorted(extra)}")
        for key, subschema in properties.items():
            if key in instance:
                _validate_against_schema(instance[key], subschema)
    elif schema_type == "array":
        if not isinstance(instance, list):
            raise ValueError(f"expected array, got {type(instance).__name__}")
        item_schema = schema.get("items")
        if item_schema:
            for item in instance:
                _validate_against_schema(item, item_schema)
    elif schema_type == "string":
        if not isinstance(instance, str):
            raise ValueError(f"expected string, got {type(instance).__name__}")
    elif schema_type == "boolean":
        if not isinstance(instance, bool):
            raise ValueError(f"expected boolean, got {type(instance).__name__}")


def _build_repair_system_prompt(original_system_prompt: str, previous_response: str, error: Exception) -> str:
    return (
        f"{original_system_prompt}\n\n"
        f"Your previous response was invalid and could not be used: {error}\n"
        f"Your previous response was:\n{previous_response}\n\n"
        "Return ONLY the JSON object matching the schema above, with no prose, "
        "explanation, or markdown code fences."
    )


class GroqOpenAiLlmClient:
    """Implements LLMClientPort. Async so rate-limit backoff (asyncio.sleep)
    never blocks the event loop — a real, not cosmetic, difference from the
    original sync api.llm_client, whose time.sleep would have stalled every
    other in-flight request under a single-process/single-worker deploy."""

    def __init__(
        self, *, provider: LlmProvider, groq_api_key: Optional[SecretStr] = None,
        openai_api_key: Optional[SecretStr] = None,
        allow_provider_fallback: bool = True, trace_hook: Optional[TraceHook] = None,
        rate_limit_backoff_seconds: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        """Provider and credentials are construction-time facts, not per-call
        ones — that is what keeps `Settings` out of `LLMClientPort`. Keys are
        held as `SecretStr` and unwrapped only in `_api_key_for`, the single
        SDK boundary, so neither repr(), vars(), nor a traceback frame carries
        one. Rotating a key or switching provider means building a new client.

        The other defaults reproduce production exactly: fallback ON, no trace
        hook. The Phase 3C generation runner sets `allow_provider_fallback=False`
        (so a rate-limit fallback can't confound a causal comparison) and a
        `trace_hook` for physical-call accounting. `rate_limit_backoff_seconds=()`
        means fail-fast: one physical attempt per provider, no sleep. Serving
        wires fail-fast (a user-facing query must not sleep 105s behind
        nginx/httpx 60s timeouts) while keeping provider fallback; offline
        evaluation keeps the default long schedule."""
        self._provider: LlmProvider = provider
        self._groq_api_key = groq_api_key
        self._openai_api_key = openai_api_key
        self._allow_provider_fallback = allow_provider_fallback
        self._trace_hook = trace_hook
        self._rate_limit_backoff_seconds = tuple(rate_limit_backoff_seconds)
        self._clients: dict[str, tuple[Optional[str], Any]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "GroqOpenAiLlmClient":
        """Composition-root convenience. `overrides` forwards the three
        non-credential keywords (allow_provider_fallback, trace_hook,
        rate_limit_backoff_seconds)."""
        return cls(
            provider=settings.llm_provider,
            groq_api_key=settings.groq_api_key,
            openai_api_key=settings.openai_api_key,
            **overrides,
        )

    def _api_key_for(self, provider: str) -> Optional[str]:
        """The single unwrap point — everything above this line handles SecretStr."""
        secret = self._groq_api_key if provider == "groq" else self._openai_api_key
        return secret.get_secret_value() if secret is not None else None

    def _emit(self, event: LlmTraceEvent) -> None:
        if self._trace_hook is not None:
            self._trace_hook(event)

    @staticmethod
    def _key_fingerprint(api_key: Optional[str]) -> Optional[str]:
        """sha256 of the key, never the key itself: the cache lives as long
        as the process and must not become a long-lived copy of secrets."""
        if api_key is None:
            return None
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    async def _sdk_client(self, provider: str, api_key: Optional[str]) -> Any:
        """One SDK client per provider, reused across calls. No `await`
        sits between the hit-path lookup and return, and rotation is
        serialized by the lock; a changed key rebuilds that provider's
        client, closing the stale one first, so rotation inside one
        instance cannot serve with stale credentials. Concurrent rotation
        is not supported (see the plan's documented limitation): serving
        passes one stable Settings and runners are sequential."""
        want = self._key_fingerprint(api_key)
        async with self._lock:
            entry = self._clients.get(provider)
            if entry is not None:
                cached_fp, cached_client = entry
                if cached_fp == want:
                    return cached_client
                await cached_client.close()
            if provider == "groq":
                created: Any = groq.AsyncGroq(api_key=api_key, max_retries=0)
            else:
                created = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
            self._clients[provider] = (want, created)
            return created

    async def aclose(self) -> None:
        """Lifespan/runner teardown. Idempotent: closing twice, or closing a
        client that never made a call, closes nothing and raises nothing.
        Every cached client is attempted even if an earlier close fails;
        failures are aggregated into an ExceptionGroup raised afterwards, so
        one broken client cannot silently leak the rest."""
        async with self._lock:
            cached = list(self._clients.values())
            self._clients.clear()
        errors: list[Exception] = []
        for _, cached_client in cached:
            try:
                await cached_client.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to close one or more LLM clients", errors)

    async def _invoke(
        self,
        provider: str,
        phase: str,
        model: str,
        schema_mode: str,
        create: Awaitable[Any],
    ) -> Any:
        """One physical provider round trip. Emits `physical_attempt` before the
        call, then exactly one of `physical_request` (success, with usage +
        latency) or `physical_failed` (any exception, with latency + error
        shape) — a 429, a schema-unsupported 400, and a network drop are all
        real physical calls and are all counted."""
        self._emit(
            LlmTraceEvent(
                event="physical_attempt", provider=provider, phase=phase, model=model, schema_mode=schema_mode
            )
        )
        start = time.monotonic()
        try:
            response = await create
        except Exception as exc:
            self._emit(
                LlmTraceEvent(
                    event="physical_failed",
                    provider=provider,
                    phase=phase,
                    model=model,
                    schema_mode=schema_mode,
                    latency_ms=(time.monotonic() - start) * 1000,
                    **_provider_error_trace_fields(exc),
                )
            )
            raise
        self._emit(
            LlmTraceEvent(
                event="physical_request",
                provider=provider,
                phase=phase,
                latency_ms=(time.monotonic() - start) * 1000,
                **_usage_trace_fields(response, model, schema_mode),
            )
        )
        return response

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        with get_tracer().start_as_current_span("llm.generate"):
            return await self._generate_structured_impl(system_prompt, user_prompt, schema)

    async def _generate_structured_impl(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        primary = self._provider
        fallback = _other_provider(primary)
        providers = (primary, fallback) if self._allow_provider_fallback else (primary,)
        attempts_summary: list[str] = []

        attempted_any = False
        for provider in providers:
            api_key = self._api_key_for(provider)
            if api_key is None:
                logger.warning(
                    "provider skipped because its API key is not configured",
                    extra={"provider": provider, "role": "primary" if provider == primary else "fallback"},
                )
                attempts_summary.append(f"{provider}: not configured")
                continue
            if attempted_any:
                self._emit(
                    LlmTraceEvent(
                        event="provider_fallback",
                        provider=provider,
                        phase="initial",
                    )
                )
            attempted_any = True
            logger.info(
                "attempting structured generation",
                extra={"provider": provider, "role": "primary" if provider == primary else "fallback"},
            )

            try:
                raw_text = await self._get_provider_response(
                    provider, system_prompt, user_prompt, schema, api_key, phase="initial"
                )
            except Exception as exc:
                logger.error(
                    "provider call failed, moving to next provider",
                    extra=_provider_error_fields(provider, exc),
                )
                self._emit(
                    LlmTraceEvent(
                        event="provider_call_failed",
                        provider=provider,
                        phase="initial",
                        **_provider_error_trace_fields(exc),
                    )
                )
                attempts_summary.append(f"{provider}: call failed ({type(exc).__name__})")
                continue

            parsed, error = _try_parse_and_validate(raw_text, schema)
            if error is None:
                assert parsed is not None
                logger.info("structured generation succeeded", extra={"provider": provider, "repaired": False})
                return parsed

            logger.warning(
                "invalid structured response, attempting repair retry",
                extra={"provider": provider, "error": str(error)},
            )
            self._emit(
                LlmTraceEvent(
                    event="repair_triggered",
                    provider=provider,
                    phase="initial",
                    error_type=type(error).__name__,
                )
            )
            repair_system_prompt = _build_repair_system_prompt(system_prompt, raw_text, error)

            try:
                repaired_raw = await self._get_provider_response(
                    provider, repair_system_prompt, user_prompt, schema, api_key, phase="repair"
                )
            except Exception as exc:
                logger.error(
                    "repair retry call failed, moving to next provider",
                    extra=_provider_error_fields(provider, exc),
                )
                self._emit(
                    LlmTraceEvent(
                        event="provider_call_failed",
                        provider=provider,
                        phase="repair",
                        **_provider_error_trace_fields(exc),
                    )
                )
                attempts_summary.append(f"{provider}: repair call failed ({type(exc).__name__})")
                continue

            parsed, repair_error = _try_parse_and_validate(repaired_raw, schema)
            if repair_error is None:
                assert parsed is not None
                logger.info(
                    "structured generation succeeded after repair", extra={"provider": provider, "repaired": True}
                )
                return parsed

            logger.warning(
                "repair retry also invalid, falling over to next provider",
                extra={"provider": provider, "error": str(repair_error)},
            )
            attempts_summary.append(f"{provider}: repair validation failed ({repair_error})")

        logger.error("structured generation failed on all providers", extra={"attempts": attempts_summary})
        self._emit(
            LlmTraceEvent(event="generation_exhausted", provider=providers[-1], phase="terminal")
        )
        raise GenerationError(
            "Structured generation failed on both providers after retries: " + "; ".join(attempts_summary)
        )

    async def _get_provider_response(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        api_key: Optional[str],
        *,
        phase: str,
    ) -> str:
        """One logical call to `provider`, with rate-limit backoff handled
        here — a separate path from JSON-repair retries, which live in
        generate_structured. Retries the SAME provider on a rate-limit
        error, up to len(RATE_LIMIT_BACKOFF_SECONDS) times, honoring
        Retry-After when the exception carries one. Re-raises once that
        allowance is exhausted."""
        last_exc: Optional[Exception] = None
        for attempt_index in range(len(self._rate_limit_backoff_seconds) + 1):
            try:
                return await self._call_provider(
                    provider, system_prompt, user_prompt, schema, api_key, phase=phase
                )
            except _RATE_LIMIT_ERROR_TYPES as exc:
                last_exc = exc
                if attempt_index >= len(self._rate_limit_backoff_seconds):
                    logger.warning(
                        "rate limit backoff exhausted", extra={"provider": provider, "attempts": attempt_index}
                    )
                    self._emit(
                        LlmTraceEvent(
                            event="rate_limit_exhausted",
                            provider=provider,
                            phase=phase,
                            attempt=attempt_index,
                            **_provider_error_trace_fields(exc),
                        )
                    )
                    raise
                wait_seconds = _extract_retry_after_seconds(exc)
                if wait_seconds is None:
                    wait_seconds = self._rate_limit_backoff_seconds[attempt_index]
                logger.warning(
                    "rate limited, backing off before retrying same provider",
                    extra={"provider": provider, "wait_seconds": wait_seconds, "attempt": attempt_index + 1},
                )
                self._emit(
                    LlmTraceEvent(
                        event="rate_limited",
                        provider=provider,
                        phase=phase,
                        attempt=attempt_index + 1,
                        wait_seconds=wait_seconds,
                        **_provider_error_trace_fields(exc),
                    )
                )
                await asyncio.sleep(wait_seconds)
        assert last_exc is not None  # pragma: no cover - loop always returns or raises above
        raise last_exc

    async def _call_provider(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        api_key: Optional[str],
        *,
        phase: str,
    ) -> str:
        if provider == "groq":
            return await self._call_groq(system_prompt, user_prompt, schema, api_key, phase=phase)
        return await self._call_openai(system_prompt, user_prompt, schema, api_key, phase=phase)

    async def _call_groq(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], api_key: Optional[str], *, phase: str
    ) -> str:
        client = await self._sdk_client("groq", api_key)
        messages = _messages(system_prompt, user_prompt)
        try:
            # messages/response_format are built dynamically from `schema` (a
            # plain JSON Schema dict, not known statically), so they can't
            # match the SDK's precise per-role TypedDict overloads — the
            # runtime shape is correct even though the static type isn't.
            response = await self._invoke(
                "groq",
                phase,
                GROQ_MODEL,
                "json_schema",
                client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "rag_response", "schema": schema, "strict": True},
                    },
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
            )
        except _JSON_SCHEMA_RETRY_ERROR_TYPES as exc:
            if not _is_unsupported_response_format_error(exc):
                raise
            logger.warning(
                "groq json_schema response_format unsupported for this model, retrying with json_object mode",
                extra={"provider": "groq", "model": GROQ_MODEL},
            )
            self._emit(
                LlmTraceEvent(
                    event="schema_fallback", provider="groq", phase=phase, model=GROQ_MODEL,
                    schema_mode="json_object",
                )
            )
            response = await self._invoke(
                "groq",
                phase,
                GROQ_MODEL,
                "json_object",
                client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
            )
        return cast(str, response.choices[0].message.content)

    async def _call_openai(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], api_key: Optional[str], *, phase: str
    ) -> str:
        client = await self._sdk_client("openai", api_key)
        # Same dynamic-schema reasoning as _call_groq above.
        response = await self._invoke(
            "openai",
            phase,
            OPENAI_MODEL,
            "json_schema",
            client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=_messages(system_prompt, user_prompt),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "rag_response", "schema": schema, "strict": True},
                },
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            ),
        )
        return cast(str, response.choices[0].message.content)
