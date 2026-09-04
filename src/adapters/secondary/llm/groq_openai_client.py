from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from pydantic import SecretStr

from src.adapters.secondary.llm.tracing import (
    LlmTraceEvent,
    TraceHook,
    _provider_error_fields,
    _provider_error_trace_fields,
    log_llm_trace,
)
from src.adapters.secondary.llm.transport import (
    _RATE_LIMIT_ERROR_TYPES,
    GROQ_MODEL,
    MAX_COMPLETION_TOKENS,
    OPENAI_MODEL,
    RATE_LIMIT_BACKOFF_SECONDS,
    ProviderTransport,
    _extract_retry_after_seconds,
)
from src.adapters.secondary.llm.validation import (
    _build_repair_system_prompt,
    _try_parse_and_validate,
)
from src.core.config import LlmProvider, Settings
from src.core.errors import GenerationError
from src.core.telemetry import get_tracer

logger = logging.getLogger(__name__)

# Re-exported so every existing import path keeps resolving after the
# 2026-09-04 split: app.lifespan, the Phase 3C runner and the test suite all
# import these from here.
__all__ = [
    "GROQ_MODEL",
    "MAX_COMPLETION_TOKENS",
    "OPENAI_MODEL",
    "RATE_LIMIT_BACKOFF_SECONDS",
    "GroqOpenAiLlmClient",
    "LlmTraceEvent",
    "TraceHook",
    "log_llm_trace",
]


def _other_provider(provider: str) -> str:
    return "openai" if provider == "groq" else "groq"


class GroqOpenAiLlmClient:
    """Implements LLMClientPort. Owns POLICY — provider failover, rate-limit
    backoff and JSON repair — over a ProviderTransport that owns the SDK
    boundary. Async so rate-limit backoff (asyncio.sleep) never blocks the
    event loop, a real difference from the original sync api.llm_client whose
    time.sleep would have stalled every other in-flight request under a
    single-process/single-worker deploy."""

    def __init__(
        self, *, provider: LlmProvider, groq_api_key: Optional[SecretStr] = None,
        openai_api_key: Optional[SecretStr] = None,
        allow_provider_fallback: bool = True, trace_hook: Optional[TraceHook] = None,
        rate_limit_backoff_seconds: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        """Provider and credentials are construction-time facts, not per-call
        ones — that is what keeps `Settings` out of `LLMClientPort`. Keys are
        held as `SecretStr` and unwrapped only in `_api_key_for`, the single
        point that hands one to the transport, so neither repr(), vars(), nor a
        traceback frame carries one. Rotating a key or switching provider means
        building a new client.

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
        self._transport = ProviderTransport(trace_hook=trace_hook)

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

    async def aclose(self) -> None:
        """Lifespan/runner teardown, delegated to the transport that owns the
        SDK clients. Idempotent, and one owner closes exactly once: app.lifespan's
        finally, or the offline runner's owning asyncio.run."""
        await self._transport.aclose()

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
                return await self._transport.call_provider(
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
