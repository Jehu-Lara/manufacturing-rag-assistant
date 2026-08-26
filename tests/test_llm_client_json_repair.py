from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import groq
import httpx
import httpx2
import openai
import pytest

from api.config import Settings
from api.llm_client import GenerationError, generate_structured
from api.prompts import JSON_SCHEMA

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


def _groq_rate_limit_error(retry_after: str | None = None) -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, request=request, headers=headers, json={"error": {"message": "rate limited"}})
    return groq.RateLimitError("rate limited", response=response, body={"error": {"message": "rate limited"}})


def _openai_rate_limit_error() -> openai.RateLimitError:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(429, request=request, json={"error": {"message": "rate limited"}})
    return openai.RateLimitError("rate limited", response=response, body={"error": {"message": "rate limited"}})


def _groq_unsupported_response_format_error() -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    message = "response_format of type json_schema is not supported for this model"
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return groq.BadRequestError(message, response=response, body={"error": {"message": message}})


@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_valid_json_first_try_returns_immediately_no_retry_no_fallback(mock_groq_cls, mock_openai_cls):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.return_value = _response(VALID_JSON_TEXT)
    mock_groq_cls.return_value = mock_groq_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    assert mock_groq_client.chat.completions.create.call_count == 1
    mock_openai_cls.assert_not_called()


@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_invalid_json_then_valid_on_repair_same_provider(mock_groq_cls, mock_openai_cls):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _response(INVALID_JSON_TEXT),
        _response(VALID_JSON_TEXT),
    ]
    mock_groq_cls.return_value = mock_groq_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    assert mock_groq_client.chat.completions.create.call_count == 2
    mock_openai_cls.assert_not_called()


@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_invalid_json_both_tries_falls_over_to_other_provider(mock_groq_cls, mock_openai_cls):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _response(INVALID_JSON_TEXT),
        _response(INVALID_JSON_TEXT),
    ]
    mock_groq_cls.return_value = mock_groq_client

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _response(VALID_JSON_TEXT)
    mock_openai_cls.return_value = mock_openai_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    assert mock_groq_client.chat.completions.create.call_count == 2
    assert mock_openai_client.chat.completions.create.call_count == 1


@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_both_providers_exhaust_repair_retries_raises_generation_error(mock_groq_cls, mock_openai_cls):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _response(INVALID_JSON_TEXT),
        _response(INVALID_JSON_TEXT),
    ]
    mock_groq_cls.return_value = mock_groq_client

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [
        _response(INVALID_JSON_TEXT),
        _response(INVALID_JSON_TEXT),
    ]
    mock_openai_cls.return_value = mock_openai_client

    with pytest.raises(GenerationError):
        generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert mock_groq_client.chat.completions.create.call_count == 2
    assert mock_openai_client.chat.completions.create.call_count == 2


@patch("api.llm_client.time.sleep")
@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_rate_limit_backs_off_and_retries_same_provider(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _groq_rate_limit_error(),
        _response(VALID_JSON_TEXT),
    ]
    mock_groq_cls.return_value = mock_groq_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    assert mock_groq_client.chat.completions.create.call_count == 2
    mock_openai_cls.assert_not_called()
    mock_sleep.assert_called_once_with(15)


@patch("api.llm_client.time.sleep")
@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_rate_limit_honors_retry_after_header(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _groq_rate_limit_error(retry_after="7"),
        _response(VALID_JSON_TEXT),
    ]
    mock_groq_cls.return_value = mock_groq_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    mock_sleep.assert_called_once_with(7.0)


@patch("api.llm_client.time.sleep")
@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_rate_limit_backoff_exhausted_falls_over_to_secondary(mock_groq_cls, mock_openai_cls, mock_sleep):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _groq_rate_limit_error(),
        _groq_rate_limit_error(),
        _groq_rate_limit_error(),
        _groq_rate_limit_error(),
    ]
    mock_groq_cls.return_value = mock_groq_client

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _response(VALID_JSON_TEXT)
    mock_openai_cls.return_value = mock_openai_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    assert mock_groq_client.chat.completions.create.call_count == 4
    assert mock_openai_client.chat.completions.create.call_count == 1
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(15)
    mock_sleep.assert_any_call(30)
    mock_sleep.assert_any_call(60)


@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_groq_unsupported_json_schema_mode_retries_with_json_object_same_attempt(mock_groq_cls, mock_openai_cls):
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = [
        _groq_unsupported_response_format_error(),
        _response(VALID_JSON_TEXT),
    ]
    mock_groq_cls.return_value = mock_groq_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))

    assert result == VALID_PAYLOAD
    assert mock_groq_client.chat.completions.create.call_count == 2
    first_call_kwargs = mock_groq_client.chat.completions.create.call_args_list[0].kwargs
    second_call_kwargs = mock_groq_client.chat.completions.create.call_args_list[1].kwargs
    assert first_call_kwargs["response_format"]["type"] == "json_schema"
    assert second_call_kwargs["response_format"] == {"type": "json_object"}
    mock_openai_cls.assert_not_called()


@patch("api.llm_client.openai.OpenAI")
@patch("api.llm_client.groq.Groq")
def test_openai_as_primary_provider_used_when_configured(mock_groq_cls, mock_openai_cls):
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _response(VALID_JSON_TEXT)
    mock_openai_cls.return_value = mock_openai_client

    result = generate_structured("system", "user", JSON_SCHEMA, _settings("openai"))

    assert result == VALID_PAYLOAD
    assert mock_openai_client.chat.completions.create.call_count == 1
    mock_groq_cls.assert_not_called()
