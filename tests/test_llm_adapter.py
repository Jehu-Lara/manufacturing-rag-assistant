from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import groq
import httpx
import pytest

from src.adapters.secondary.llm.groq_openai_client import MAX_COMPLETION_TOKENS, GroqOpenAiLlmClient
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
