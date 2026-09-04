from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Awaitable, Optional, cast

import groq
import openai

from src.adapters.secondary.llm.tracing import (
    LlmTraceEvent,
    TraceHook,
    _provider_error_trace_fields,
    _usage_trace_fields,
)

logger = logging.getLogger(__name__)


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


class ProviderTransport:
    """The single SDK boundary: the only place `groq` and `openai` are imported
    and called, so the test suite has exactly one place to patch. Owns the
    per-provider client cache and its teardown, and emits the physical-call
    trace events. Failover, rate-limit backoff and JSON repair are policy and
    live in GroqOpenAiLlmClient, not here."""

    def __init__(self, *, trace_hook: Optional[TraceHook] = None) -> None:
        self._trace_hook = trace_hook
        self._clients: dict[str, tuple[Optional[str], Any]] = {}
        self._lock = asyncio.Lock()

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

    async def sdk_client(self, provider: str, api_key: Optional[str]) -> Any:
        """One SDK client per provider, reused across calls. No `await` sits
        between the hit-path lookup and return, and creation is serialized by
        the lock. Since the owning client's credentials became immutable
        (Bucket 2), the fingerprint branch guards against in-place mutation
        rather than serving a rotation path — rotation means a new client."""
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

    async def call_provider(
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
        client = await self.sdk_client("groq", api_key)
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
        client = await self.sdk_client("openai", api_key)
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
