from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, cast

import groq
import openai

from src.core.config import Settings
from src.core.errors import GenerationError

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

# Fixed backoff schedule (seconds) used when a provider's rate-limit error
# carries no Retry-After value. Index 0 is the wait before the 1st retry,
# etc. len() also defines the max number of rate-limit retries per call.
RATE_LIMIT_BACKOFF_SECONDS = (15, 30, 60)

_RATE_LIMIT_ERROR_TYPES = (groq.RateLimitError, openai.RateLimitError)
_JSON_SCHEMA_RETRY_ERROR_TYPES = (groq.BadRequestError, groq.UnprocessableEntityError)


def _other_provider(provider: str) -> str:
    return "openai" if provider == "groq" else "groq"


def _api_key_for(provider: str, settings: Settings) -> Optional[str]:
    return settings.groq_api_key if provider == "groq" else settings.openai_api_key


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

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], settings: Settings
    ) -> dict[str, Any]:
        primary = settings.llm_provider
        fallback = _other_provider(primary)
        attempts_summary: list[str] = []

        for provider in (primary, fallback):
            api_key = _api_key_for(provider, settings)
            logger.info(
                "attempting structured generation",
                extra={"provider": provider, "role": "primary" if provider == primary else "fallback"},
            )

            try:
                raw_text = await self._get_provider_response(provider, system_prompt, user_prompt, schema, api_key)
            except Exception as exc:
                logger.error(
                    "provider call failed, moving to next provider",
                    extra={"provider": provider, "error": str(exc)},
                )
                attempts_summary.append(f"{provider}: call failed ({exc})")
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
            repair_system_prompt = _build_repair_system_prompt(system_prompt, raw_text, error)

            try:
                repaired_raw = await self._get_provider_response(
                    provider, repair_system_prompt, user_prompt, schema, api_key
                )
            except Exception as exc:
                logger.error(
                    "repair retry call failed, moving to next provider",
                    extra={"provider": provider, "error": str(exc)},
                )
                attempts_summary.append(f"{provider}: repair call failed ({exc})")
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
        raise GenerationError(
            "Structured generation failed on both providers after retries: " + "; ".join(attempts_summary)
        )

    async def _get_provider_response(
        self, provider: str, system_prompt: str, user_prompt: str, schema: dict[str, Any], api_key: Optional[str]
    ) -> str:
        """One logical call to `provider`, with rate-limit backoff handled
        here — a separate path from JSON-repair retries, which live in
        generate_structured. Retries the SAME provider on a rate-limit
        error, up to len(RATE_LIMIT_BACKOFF_SECONDS) times, honoring
        Retry-After when the exception carries one. Re-raises once that
        allowance is exhausted."""
        last_exc: Optional[Exception] = None
        for attempt_index in range(len(RATE_LIMIT_BACKOFF_SECONDS) + 1):
            try:
                return await self._call_provider(provider, system_prompt, user_prompt, schema, api_key)
            except _RATE_LIMIT_ERROR_TYPES as exc:
                last_exc = exc
                if attempt_index >= len(RATE_LIMIT_BACKOFF_SECONDS):
                    logger.warning(
                        "rate limit backoff exhausted", extra={"provider": provider, "attempts": attempt_index}
                    )
                    raise
                wait_seconds = _extract_retry_after_seconds(exc)
                if wait_seconds is None:
                    wait_seconds = RATE_LIMIT_BACKOFF_SECONDS[attempt_index]
                logger.warning(
                    "rate limited, backing off before retrying same provider",
                    extra={"provider": provider, "wait_seconds": wait_seconds, "attempt": attempt_index + 1},
                )
                await asyncio.sleep(wait_seconds)
        assert last_exc is not None  # pragma: no cover - loop always returns or raises above
        raise last_exc

    async def _call_provider(
        self, provider: str, system_prompt: str, user_prompt: str, schema: dict[str, Any], api_key: Optional[str]
    ) -> str:
        if provider == "groq":
            return await self._call_groq(system_prompt, user_prompt, schema, api_key)
        return await self._call_openai(system_prompt, user_prompt, schema, api_key)

    async def _call_groq(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], api_key: Optional[str]
    ) -> str:
        client = groq.AsyncGroq(api_key=api_key, max_retries=0)
        messages = _messages(system_prompt, user_prompt)
        try:
            # messages/response_format are built dynamically from `schema` (a
            # plain JSON Schema dict, not known statically), so they can't
            # match the SDK's precise per-role TypedDict overloads — the
            # runtime shape is correct even though the static type isn't.
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "rag_response", "schema": schema, "strict": True},
                },
            )  # type: ignore[call-overload]
        except _JSON_SCHEMA_RETRY_ERROR_TYPES as exc:
            if not _is_unsupported_response_format_error(exc):
                raise
            logger.warning(
                "groq json_schema response_format unsupported for this model, retrying with json_object mode",
                extra={"provider": "groq", "model": GROQ_MODEL},
            )
            response = await client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, response_format={"type": "json_object"}
            )  # type: ignore[call-overload]
        return cast(str, response.choices[0].message.content)

    async def _call_openai(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], api_key: Optional[str]
    ) -> str:
        client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        # Same dynamic-schema reasoning as _call_groq above.
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=_messages(system_prompt, user_prompt),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rag_response", "schema": schema, "strict": True},
            },
        )  # type: ignore[call-overload]
        return cast(str, response.choices[0].message.content)
