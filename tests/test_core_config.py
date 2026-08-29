from __future__ import annotations

import pytest

from src.core.config import Settings, load_settings


def test_default_refusal_threshold_is_0_5999(monkeypatch):
    monkeypatch.delenv("REFUSAL_COSINE_THRESHOLD", raising=False)
    settings = load_settings()
    assert settings.refusal_cosine_threshold == 0.5999


def test_invalid_llm_provider_raises_value_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings()


def test_non_float_threshold_raises_value_error(monkeypatch):
    monkeypatch.setenv("REFUSAL_COSINE_THRESHOLD", "not-a-float")
    with pytest.raises(ValueError, match="REFUSAL_COSINE_THRESHOLD"):
        load_settings()


def test_non_int_rate_limit_raises_value_error(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "not-an-int")
    with pytest.raises(ValueError, match="RATE_LIMIT_PER_MINUTE"):
        load_settings()


def test_direct_construction_ignores_ambient_env_for_omitted_fields(monkeypatch):
    """Settings is a plain pydantic BaseModel, not pydantic_settings.BaseSettings,
    specifically so a directly-constructed instance (as ~15 existing tests do)
    never silently picks up an ambient env var for a field it didn't pass —
    verified directly that BaseSettings does not have this property."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "999")
    settings = Settings(
        groq_api_key="fake-key",
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=0.3,
        log_level="INFO",
    )
    assert settings.rate_limit_per_minute == 20


def test_settings_is_frozen():
    settings = Settings(
        groq_api_key=None,
        openai_api_key=None,
        llm_provider="groq",
        refusal_cosine_threshold=0.3,
        log_level="INFO",
    )
    with pytest.raises(Exception):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_settings_repr_redacts_all_secret_values():
    settings = Settings(
        groq_api_key="groq-secret",
        openai_api_key="openai-secret",
        llm_provider="groq",
        refusal_cosine_threshold=0.3,
        log_level="INFO",
        api_key="internal-secret",
    )

    rendered = repr(settings)
    for secret in ("groq-secret", "openai-secret", "internal-secret"):
        assert secret not in rendered
    assert rendered.count("**********") == 3


def test_default_cors_origins_are_empty(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    assert load_settings().cors_allow_origins == []
