from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

LlmProvider = Literal["groq", "openai"]

# Overridden from the threshold_analysis.py-selected 0.5599 — see SPEC.md's
# Phase 3 status note for why (Step 2's tie-break objective and Step 6's
# acceptance targets disagreed; 0.5999 is closer to both downstream targets).
_DEFAULT_REFUSAL_COSINE_THRESHOLD = 0.5999
_DEFAULT_LLM_PROVIDER: LlmProvider = "groq"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_RATE_LIMIT_PER_MINUTE = 20
_DEFAULT_CORS_ALLOW_ORIGINS: tuple[str, ...] = ()

# Repo root, used only to compute default index paths below when CHROMA_PATH/
# BM25_PATH aren't set — same physical retrieval/output/ location the old
# retrieval/build_index.py always wrote to, so an already-built index keeps
# working through the src/ migration without requiring new env vars. The
# BM25 filename is .json, not .pkl: see src/adapters/secondary/lexical (no
# pickle in runtime code, ADR-004).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHROMA_PATH = _REPO_ROOT / "retrieval" / "output" / "chroma"
_DEFAULT_BM25_PATH = _REPO_ROOT / "retrieval" / "output" / "bm25_index.json"


class Settings(BaseModel):
    """A validated, inert data holder — not a pydantic_settings.BaseSettings.
    BaseSettings consults the process environment for any field omitted at
    construction time (verified directly: it overrides a field's coded
    default with an ambient env var even when that field isn't passed to
    the constructor), which would silently contaminate the many existing
    tests that construct Settings(...) directly without passing every
    field. load_settings() below is the sole place environment variables
    are read, exactly as CLAUDE.md's config/env-loading pattern specifies."""

    model_config = {"frozen": True}

    groq_api_key: Optional[SecretStr]
    openai_api_key: Optional[SecretStr]
    llm_provider: LlmProvider
    refusal_cosine_threshold: float
    log_level: str
    rate_limit_per_minute: int = _DEFAULT_RATE_LIMIT_PER_MINUTE
    api_key: Optional[SecretStr] = None
    cors_allow_origins: list[str] = list(_DEFAULT_CORS_ALLOW_ORIGINS)
    chroma_path: Path = _DEFAULT_CHROMA_PATH
    bm25_path: Path = _DEFAULT_BM25_PATH


def load_settings() -> Settings:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    groq_api_key_raw = os.environ.get("GROQ_API_KEY") or None
    groq_api_key = SecretStr(groq_api_key_raw) if groq_api_key_raw is not None else None
    openai_api_key_raw = os.environ.get("OPENAI_API_KEY") or None
    openai_api_key = SecretStr(openai_api_key_raw) if openai_api_key_raw is not None else None

    llm_provider_raw = os.environ.get("LLM_PROVIDER", _DEFAULT_LLM_PROVIDER)
    if llm_provider_raw not in ("groq", "openai"):
        raise ValueError(f"LLM_PROVIDER must be 'groq' or 'openai', got {llm_provider_raw!r}")
    llm_provider: LlmProvider = llm_provider_raw  # type: ignore[assignment]

    threshold_raw = os.environ.get("REFUSAL_COSINE_THRESHOLD")
    if threshold_raw is None:
        refusal_cosine_threshold = _DEFAULT_REFUSAL_COSINE_THRESHOLD
    else:
        try:
            refusal_cosine_threshold = float(threshold_raw)
        except ValueError as exc:
            raise ValueError(f"REFUSAL_COSINE_THRESHOLD must be a float, got {threshold_raw!r}") from exc

    log_level = os.environ.get("LOG_LEVEL", _DEFAULT_LOG_LEVEL)

    rate_limit_raw = os.environ.get("RATE_LIMIT_PER_MINUTE")
    if rate_limit_raw is None:
        rate_limit_per_minute = _DEFAULT_RATE_LIMIT_PER_MINUTE
    else:
        try:
            rate_limit_per_minute = int(rate_limit_raw)
        except ValueError as exc:
            raise ValueError(f"RATE_LIMIT_PER_MINUTE must be an int, got {rate_limit_raw!r}") from exc

    api_key_raw = os.environ.get("API_KEY") or None
    api_key = SecretStr(api_key_raw) if api_key_raw is not None else None

    cors_raw = os.environ.get("CORS_ALLOW_ORIGINS")
    cors_allow_origins = (
        [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
        if cors_raw
        else list(_DEFAULT_CORS_ALLOW_ORIGINS)
    )

    chroma_path_raw = os.environ.get("CHROMA_PATH")
    chroma_path = Path(chroma_path_raw) if chroma_path_raw else _DEFAULT_CHROMA_PATH

    bm25_path_raw = os.environ.get("BM25_PATH")
    bm25_path = Path(bm25_path_raw) if bm25_path_raw else _DEFAULT_BM25_PATH

    return Settings(
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        llm_provider=llm_provider,
        refusal_cosine_threshold=refusal_cosine_threshold,
        log_level=log_level,
        rate_limit_per_minute=rate_limit_per_minute,
        api_key=api_key,
        cors_allow_origins=cors_allow_origins,
        chroma_path=chroma_path,
        bm25_path=bm25_path,
    )
