from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import groq
import httpx
import pytest

from src.adapters.secondary.llm.groq_openai_client import (
    MAX_COMPLETION_TOKENS,
    GroqOpenAiLlmClient,
    log_llm_trace,
)
from src.core.config import Settings
from src.core.errors import GenerationError
from src.features.query.prompts import JSON_SCHEMA

VALID_PAYLOAD = {"answer": "answer text", "citations": [{"chunk_id": "chunk-1"}], "refused": False}
VALID_JSON_TEXT = json.dumps(VALID_PAYLOAD)
INVALID_JSON_TEXT = "this is not valid json {"


def _settings(provider: str = "groq") -> Settings:
    return Settings(
        groq_api_key="groq-test-key",
        openai_api_key="openai-test-key",
        llm_provider=provider,
        refusal_cosine_threshold=0.5599,
        log_level="INFO",
    )


def _response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _async_client_with_create(*side_effects_or_return: object, side_effect: bool = False) -> MagicMock:
    """Builds a MagicMock client whose .chat.completions.create is an
    AsyncMock (awaitable) — the rest of the client object stays a plain
    MagicMock since only .create is ever awaited."""
    client = MagicMock()
    create = AsyncMock()
    if side_effect:
        create.side_effect = list(side_effects_or_return)
    else:
        create.return_value = side_effects_or_return[0]
    client.chat.completions.create = create
    return client


def _groq_rate_limit_error(retry_after: str | None = None) -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, request=request, headers=headers, json={"error": {"message": "rate limited"}})
    return groq.RateLimitError("rate limited", response=response, body={"error": {"message": "rate limited"}})


def _groq_unsupported_response_format_error() -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    message = "response_format of type json_schema is not supported for this model"
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return groq.BadRequestError(message, response=response, body={"error": {"message": message}})


def _run(coro):
    return asyncio.run(coro)


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_valid_json_first_try_returns_immediately_no_retry_no_fallback(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    assert mock_groq_cls.return_value.chat.completions.create.await_count == 1
    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_invalid_json_then_valid_on_repair_same_provider(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _response(INVALID_JSON_TEXT), _response(VALID_JSON_TEXT), side_effect=True
    )

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    assert mock_groq_cls.return_value.chat.completions.create.await_count == 2
    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_invalid_json_both_tries_falls_over_to_other_provider(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _response(INVALID_JSON_TEXT), _response(INVALID_JSON_TEXT), side_effect=True
    )
    mock_openai_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    assert mock_groq_cls.return_value.chat.completions.create.await_count == 2
    assert mock_openai_cls.return_value.chat.completions.create.await_count == 1


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_both_providers_exhaust_repair_retries_raises_generation_error(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _response(INVALID_JSON_TEXT), _response(INVALID_JSON_TEXT), side_effect=True
    )
    mock_openai_cls.return_value = _async_client_with_create(
        _response(INVALID_JSON_TEXT), _response(INVALID_JSON_TEXT), side_effect=True
    )

    with pytest.raises(GenerationError):
        _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert mock_groq_cls.return_value.chat.completions.create.await_count == 2
    assert mock_openai_cls.return_value.chat.completions.create.await_count == 2


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_rate_limit_backs_off_via_asyncio_sleep_and_retries_same_provider(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_rate_limit_error(), _response(VALID_JSON_TEXT), side_effect=True
    )

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    assert mock_groq_cls.return_value.chat.completions.create.await_count == 2
    mock_openai_cls.assert_not_called()
    mock_sleep.assert_called_once_with(15)


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_rate_limit_honors_retry_after_header(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_rate_limit_error(retry_after="7"), _response(VALID_JSON_TEXT), side_effect=True
    )

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    mock_sleep.assert_called_once_with(7.0)


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_rate_limit_backoff_exhausted_falls_over_to_secondary(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_rate_limit_error(), _groq_rate_limit_error(), _groq_rate_limit_error(), _groq_rate_limit_error(),
        side_effect=True,
    )
    mock_openai_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    assert mock_groq_cls.return_value.chat.completions.create.await_count == 4
    assert mock_openai_cls.return_value.chat.completions.create.await_count == 1
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(15)
    mock_sleep.assert_any_call(30)
    mock_sleep.assert_any_call(60)


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_groq_unsupported_json_schema_mode_retries_with_json_object_same_attempt(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_unsupported_response_format_error(), _response(VALID_JSON_TEXT), side_effect=True
    )

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert result == VALID_PAYLOAD
    create_mock = mock_groq_cls.return_value.chat.completions.create
    assert create_mock.await_count == 2
    first_call_kwargs = create_mock.call_args_list[0].kwargs
    second_call_kwargs = create_mock.call_args_list[1].kwargs
    assert first_call_kwargs["response_format"]["type"] == "json_schema"
    assert second_call_kwargs["response_format"] == {"type": "json_object"}
    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_openai_as_primary_provider_used_when_configured(mock_groq_cls, mock_openai_cls):
    mock_openai_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    result = _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("openai")))

    assert result == VALID_PAYLOAD
    assert mock_openai_cls.return_value.chat.completions.create.await_count == 1
    mock_groq_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_provider_calls_enforce_completion_token_limit(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _response(INVALID_JSON_TEXT), _response(INVALID_JSON_TEXT), side_effect=True
    )
    mock_openai_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    for call in mock_groq_cls.return_value.chat.completions.create.await_args_list:
        assert call.kwargs["max_completion_tokens"] == MAX_COMPLETION_TOKENS
    assert mock_openai_cls.return_value.chat.completions.create.await_args.kwargs["max_completion_tokens"] == (
        MAX_COMPLETION_TOKENS
    )


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_unconfigured_fallback_is_skipped_instead_of_receiving_none(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(RuntimeError("provider failed"), side_effect=True)
    settings = _settings("groq").model_copy(update={"openai_api_key": None})

    with pytest.raises(GenerationError):
        _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, settings))

    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_provider_exception_text_is_not_logged_or_returned(mock_groq_cls, mock_openai_cls, caplog):
    sensitive_text = "request echoed confidential-question"
    mock_groq_cls.return_value = _async_client_with_create(RuntimeError(sensitive_text), side_effect=True)
    settings = _settings("groq").model_copy(update={"openai_api_key": None})

    with caplog.at_level("ERROR"), pytest.raises(GenerationError) as exc_info:
        _run(GroqOpenAiLlmClient().generate_structured("system", "user", JSON_SCHEMA, settings))

    assert sensitive_text not in caplog.text
    assert sensitive_text not in str(exc_info.value)


# --- Phase 3C: allow_provider_fallback=False + trace_hook ---


def _response_with_usage(content: str, *, finish_reason: str = "stop") -> MagicMock:
    response = _response(content)
    response.choices[0].finish_reason = finish_reason
    response.usage.prompt_tokens = 111
    response.usage.completion_tokens = 22
    response.usage.total_tokens = 133
    response.system_fingerprint = "fp_test"
    return response


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_allow_provider_fallback_false_never_calls_the_other_provider(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _response(INVALID_JSON_TEXT), _response(INVALID_JSON_TEXT), side_effect=True
    )
    mock_openai_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    with pytest.raises(GenerationError):
        _run(
            GroqOpenAiLlmClient(allow_provider_fallback=False).generate_structured(
                "system", "user", JSON_SCHEMA, _settings("groq")
            )
        )

    assert mock_groq_cls.return_value.chat.completions.create.await_count == 2
    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_trace_hook_records_content_free_physical_request(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(_response_with_usage(VALID_JSON_TEXT))
    events = []

    _run(
        GroqOpenAiLlmClient(trace_hook=events.append).generate_structured(
            "SENSITIVE-SYSTEM", "SENSITIVE-USER", JSON_SCHEMA, _settings("groq")
        )
    )

    physical = [e for e in events if e.event == "physical_request"]
    assert len(physical) == 1
    ev = physical[0]
    assert ev.provider == "groq" and ev.phase == "initial"
    assert ev.prompt_tokens == 111 and ev.completion_tokens == 22 and ev.total_tokens == 133
    assert ev.finish_reason == "stop" and ev.system_fingerprint == "fp_test"
    blob = repr(ev)
    assert "SENSITIVE" not in blob and "groq-test-key" not in blob and VALID_JSON_TEXT not in blob


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_trace_hook_records_rate_limit_and_repair_events(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_rate_limit_error(), _response(INVALID_JSON_TEXT), _response(VALID_JSON_TEXT), side_effect=True
    )
    events = []

    _run(
        GroqOpenAiLlmClient(allow_provider_fallback=False, trace_hook=events.append).generate_structured(
            "system", "user", JSON_SCHEMA, _settings("groq")
        )
    )

    names = [e.event for e in events]
    assert "rate_limited" in names
    assert "repair_triggered" in names
    assert [e for e in events if e.event == "physical_request" and e.phase == "repair"]


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_trace_hook_counts_failed_physical_attempts(mock_groq_cls, mock_openai_cls, mock_sleep):
    # every groq attempt 429s (4 physical attempts, all failed), then openai succeeds
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_rate_limit_error(), _groq_rate_limit_error(), _groq_rate_limit_error(), _groq_rate_limit_error(),
        side_effect=True,
    )
    mock_openai_cls.return_value = _async_client_with_create(_response_with_usage(VALID_JSON_TEXT))
    events = []

    _run(
        GroqOpenAiLlmClient(trace_hook=events.append).generate_structured(
            "system", "user", JSON_SCHEMA, _settings("groq")
        )
    )

    attempts = [e for e in events if e.event == "physical_attempt"]
    failed = [e for e in events if e.event == "physical_failed"]
    succeeded = [e for e in events if e.event == "physical_request"]
    fallbacks = [e for e in events if e.event == "provider_fallback"]
    assert len(attempts) == 5  # 4 groq + 1 openai
    assert len(failed) == 4 and all(e.provider == "groq" for e in failed)
    assert len(succeeded) == 1 and succeeded[0].provider == "openai"
    assert len(fallbacks) == 1 and fallbacks[0].provider == "openai"
    assert all(e.latency_ms is not None for e in failed + succeeded)


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_trace_hook_counts_schema_fallback_as_two_attempts(mock_groq_cls, mock_openai_cls):
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_unsupported_response_format_error(), _response_with_usage(VALID_JSON_TEXT), side_effect=True
    )
    events = []

    _run(
        GroqOpenAiLlmClient(allow_provider_fallback=False, trace_hook=events.append).generate_structured(
            "system", "user", JSON_SCHEMA, _settings("groq")
        )
    )

    assert len([e for e in events if e.event == "physical_attempt"]) == 2
    assert len([e for e in events if e.event == "physical_failed"]) == 1
    assert len([e for e in events if e.event == "schema_fallback"]) == 1
    assert len([e for e in events if e.event == "physical_request"]) == 1


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_production_trace_sink_logs_call_shape_without_any_content(
    mock_groq_cls, mock_openai_cls, mock_sleep, caplog
):
    """log_llm_trace is the hook production wires in, so what it writes to
    stdout is a real disclosure surface. Exercised over the noisiest path
    available (rate limit, then a repair retry) so every event kind gets
    emitted through it, not just the happy one."""
    sensitive_key = "groq-test-key"
    mock_groq_cls.return_value = _async_client_with_create(
        _groq_rate_limit_error(), _response(INVALID_JSON_TEXT), _response_with_usage(VALID_JSON_TEXT),
        side_effect=True,
    )

    with caplog.at_level("INFO", logger="src.adapters.secondary.llm.groq_openai_client"):
        _run(
            GroqOpenAiLlmClient(allow_provider_fallback=False, trace_hook=log_llm_trace).generate_structured(
                "SENSITIVE-SYSTEM-PROMPT", "SENSITIVE-USER-QUESTION", JSON_SCHEMA, _settings("groq")
            )
        )

    trace_records = [r for r in caplog.records if r.__dict__.get("event") == "llm_trace"]
    assert trace_records, "the production trace sink emitted nothing"
    assert {r.__dict__["llm_event"] for r in trace_records} >= {
        "physical_attempt",
        "physical_request",
        "rate_limited",
    }

    forbidden = (
        "SENSITIVE-SYSTEM-PROMPT",
        "SENSITIVE-USER-QUESTION",
        sensitive_key,
        "openai-test-key",
        "Authorization",
        VALID_JSON_TEXT,
        INVALID_JSON_TEXT,
        VALID_PAYLOAD["answer"],
    )
    blob = "\n".join(
        f"{record.getMessage()} {sorted(record.__dict__.items(), key=lambda kv: kv[0])!r}"
        for record in trace_records
    )
    for secret in forbidden:
        assert secret not in blob, f"{secret!r} leaked into an llm_trace log line"


def test_production_trace_sink_drops_unset_fields_instead_of_logging_nulls(caplog):
    from src.adapters.secondary.llm.groq_openai_client import LlmTraceEvent

    with caplog.at_level("INFO", logger="src.adapters.secondary.llm.groq_openai_client"):
        log_llm_trace(LlmTraceEvent(event="physical_attempt", provider="groq", phase="initial"))

    record = next(r for r in caplog.records if r.__dict__.get("event") == "llm_trace")
    assert record.__dict__["llm_event"] == "physical_attempt"
    assert record.__dict__["llm_provider"] == "groq"
    assert record.__dict__["llm_phase"] == "initial"
    assert "llm_latency_ms" not in record.__dict__
    assert "llm_error_type" not in record.__dict__


# A provider's error body can echo the prompt back, so str(exc) is the one
# field most likely to leak content through an error path. This marker exists
# only to make that leak detectable: it is planted in the exception message and
# body, and must appear in neither the trace events nor their log lines.
EXCEPTION_CANARY = "EXC-CANARY-4f21a9-provider-echoed-the-question-back"


def _groq_rate_limit_error_carrying_the_canary() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": EXCEPTION_CANARY}}
    response = httpx.Response(429, request=request, json=body)
    return groq.RateLimitError(EXCEPTION_CANARY, response=response, body=body)


def _recording_and_logging_hook(events: list) -> object:
    def hook(event) -> None:
        events.append(event)
        log_llm_trace(event)

    return hook


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_rate_limit_error_text_reaches_neither_the_trace_event_nor_its_log_line(
    mock_groq_cls, mock_openai_cls, mock_sleep, caplog
):
    error = _groq_rate_limit_error_carrying_the_canary()
    assert EXCEPTION_CANARY in str(error), "the marker must really be in the exception"
    mock_groq_cls.return_value = _async_client_with_create(*([error] * 5), side_effect=True)
    events: list = []

    with caplog.at_level("INFO", logger="src.adapters.secondary.llm.groq_openai_client"):
        with pytest.raises(GenerationError):
            _run(
                GroqOpenAiLlmClient(
                    allow_provider_fallback=False, trace_hook=_recording_and_logging_hook(events)
                ).generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))
            )

    # The shape IS captured — this is not passing by emitting nothing.
    assert any(e.event == "rate_limited" for e in events)
    assert any(e.error_type == "RateLimitError" for e in events)
    assert any(e.status_code == 429 for e in events)

    for event in events:
        assert EXCEPTION_CANARY not in repr(event)
    trace_records = [r for r in caplog.records if r.__dict__.get("event") == "llm_trace"]
    assert trace_records
    for record in trace_records:
        assert EXCEPTION_CANARY not in f"{record.getMessage()} {record.__dict__!r}"


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_unexpected_exception_text_reaches_neither_the_trace_event_nor_its_log_line(
    mock_groq_cls, mock_openai_cls, caplog
):
    """The non-provider path: an arbitrary exception carries no HTTP response,
    so only its type should survive into the trace."""
    mock_groq_cls.return_value = _async_client_with_create(RuntimeError(EXCEPTION_CANARY), side_effect=True)
    events: list = []

    with caplog.at_level("INFO", logger="src.adapters.secondary.llm.groq_openai_client"):
        with pytest.raises(GenerationError):
            _run(
                GroqOpenAiLlmClient(
                    allow_provider_fallback=False, trace_hook=_recording_and_logging_hook(events)
                ).generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))
            )

    assert any(e.event == "physical_failed" for e in events)
    assert any(e.error_type == "RuntimeError" for e in events)
    assert all(e.status_code is None for e in events)

    for event in events:
        assert EXCEPTION_CANARY not in repr(event)
    trace_records = [r for r in caplog.records if r.__dict__.get("event") == "llm_trace"]
    assert trace_records
    for record in trace_records:
        assert EXCEPTION_CANARY not in f"{record.getMessage()} {record.__dict__!r}"
