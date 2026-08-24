from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

LlmProvider = Literal["groq", "openai"]

_DEFAULT_REFUSAL_COSINE_THRESHOLD = 0.5599
_DEFAULT_LLM_PROVIDER: LlmProvider = "groq"
_DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str]
    openai_api_key: Optional[str]
    llm_provider: LlmProvider
    refusal_cosine_threshold: float
    log_level: str


def load_settings() -> Settings:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    groq_api_key = os.environ.get("GROQ_API_KEY") or None
    openai_api_key = os.environ.get("OPENAI_API_KEY") or None

    llm_provider_raw = os.environ.get("LLM_PROVIDER", _DEFAULT_LLM_PROVIDER)
    if llm_provider_raw not in ("groq", "openai"):
        raise ValueError(
            f"LLM_PROVIDER must be 'groq' or 'openai', got {llm_provider_raw!r}"
        )
    llm_provider: LlmProvider = llm_provider_raw  # type: ignore[assignment]

    threshold_raw = os.environ.get("REFUSAL_COSINE_THRESHOLD")
    if threshold_raw is None:
        refusal_cosine_threshold = _DEFAULT_REFUSAL_COSINE_THRESHOLD
    else:
        try:
            refusal_cosine_threshold = float(threshold_raw)
        except ValueError as exc:
            raise ValueError(
                f"REFUSAL_COSINE_THRESHOLD must be a float, got {threshold_raw!r}"
            ) from exc

    log_level = os.environ.get("LOG_LEVEL", _DEFAULT_LOG_LEVEL)

    return Settings(
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        llm_provider=llm_provider,
        refusal_cosine_threshold=refusal_cosine_threshold,
        log_level=log_level,
    )
