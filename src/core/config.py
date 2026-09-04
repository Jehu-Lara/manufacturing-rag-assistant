from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

from src.core.paths import REPO_ROOT, RETRIEVAL_OUTPUT_DIR

LlmProvider = Literal["groq", "openai"]
RefusalPolicyName = Literal["binary", "grounded_review"]
# Structurally identical to src.domain.models.IndexProfile, deliberately not
# shared: src/core must not import src/domain. tests/test_core_config.py pins
# the two together so they cannot drift.
IndexProfileName = Literal["raw-v1", "contextual-v1"]

# Overridden from the threshold_analysis.py-selected 0.5599 — see SPEC.md's
# Phase 3 status note for why (Step 2's tie-break objective and Step 6's
# acceptance targets disagreed; 0.5999 is closer to both downstream targets).
_DEFAULT_REFUSAL_COSINE_THRESHOLD = 0.5999
# Public alias — importable by report/provenance tooling that must not reach
# for a private name (see src/features/evaluation/artifacts.py).
DEFAULT_REFUSAL_COSINE_THRESHOLD = _DEFAULT_REFUSAL_COSINE_THRESHOLD

# Phase 3 (ADR-009). binary = today's single-cutoff gate. grounded_review adds
# a middle band [review_floor, cosine_threshold) that makes one verified LLM
# call instead of refusing outright. 0.5500 is pre-registered — see ADR-009 for
# the derivation from min(r001, r002) top1_semantic. Default stays binary.
_DEFAULT_REFUSAL_POLICY: RefusalPolicyName = "binary"
_DEFAULT_REFUSAL_REVIEW_FLOOR = 0.5500
DEFAULT_REFUSAL_REVIEW_FLOOR = _DEFAULT_REFUSAL_REVIEW_FLOOR

_DEFAULT_LLM_PROVIDER: LlmProvider = "groq"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_RATE_LIMIT_PER_MINUTE = 20
_DEFAULT_CORS_ALLOW_ORIGINS: tuple[str, ...] = ()

# contextual-v1 is the shipped default (ADR-008); raw-v1 is the tested rollback
# path, which is why this one IS env-overridable — unlike expansion_mode, which
# is deliberately not a Settings field so production stays hard-wired to "off".
_VALID_INDEX_PROFILES: tuple[IndexProfileName, ...] = ("raw-v1", "contextual-v1")
_DEFAULT_INDEX_PROFILE: IndexProfileName = "contextual-v1"

# Default index paths, used only when CHROMA_PATH/BM25_PATH aren't set — same
# physical retrieval/output/ location the old retrieval/build_index.py always
# wrote to, so an already-built index keeps working through the src/ migration
# without requiring new env vars. The BM25 filename is .json, not .pkl: see
# src/adapters/secondary/lexical (no pickle in runtime code, ADR-004).
_DEFAULT_CHROMA_PATH = RETRIEVAL_OUTPUT_DIR / "chroma"
_DEFAULT_BM25_PATH = RETRIEVAL_OUTPUT_DIR / "bm25_index.json"


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
    refusal_policy: RefusalPolicyName = _DEFAULT_REFUSAL_POLICY
    refusal_review_floor: float = _DEFAULT_REFUSAL_REVIEW_FLOOR
    log_level: str
    rate_limit_per_minute: int = _DEFAULT_RATE_LIMIT_PER_MINUTE
    api_key: Optional[SecretStr] = None
    cors_allow_origins: list[str] = list(_DEFAULT_CORS_ALLOW_ORIGINS)
    chroma_path: Path = _DEFAULT_CHROMA_PATH
    bm25_path: Path = _DEFAULT_BM25_PATH
    index_profile: IndexProfileName = _DEFAULT_INDEX_PROFILE
    deployed_sha: Optional[str] = None
    otlp_endpoint: Optional[str] = None


def load_settings() -> Settings:
    env_path = REPO_ROOT / ".env"
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

    refusal_policy_raw = os.environ.get("REFUSAL_POLICY", _DEFAULT_REFUSAL_POLICY)
    if refusal_policy_raw not in ("binary", "grounded_review"):
        raise ValueError(
            f"REFUSAL_POLICY must be 'binary' or 'grounded_review', got {refusal_policy_raw!r}"
        )
    refusal_policy: RefusalPolicyName = refusal_policy_raw  # type: ignore[assignment]

    review_floor_raw = os.environ.get("REFUSAL_REVIEW_FLOOR")
    if review_floor_raw is None:
        refusal_review_floor = _DEFAULT_REFUSAL_REVIEW_FLOOR
    else:
        try:
            refusal_review_floor = float(review_floor_raw)
        except ValueError as exc:
            raise ValueError(
                f"REFUSAL_REVIEW_FLOOR must be a float, got {review_floor_raw!r}"
            ) from exc

    for name, value in (
        ("REFUSAL_COSINE_THRESHOLD", refusal_cosine_threshold),
        ("REFUSAL_REVIEW_FLOOR", refusal_review_floor),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite value in [0, 1], got {value!r}")
    if refusal_policy == "grounded_review" and not refusal_review_floor < refusal_cosine_threshold:
        raise ValueError(
            "REFUSAL_REVIEW_FLOOR must be strictly below REFUSAL_COSINE_THRESHOLD when "
            f"REFUSAL_POLICY='grounded_review' (floor={refusal_review_floor}, "
            f"threshold={refusal_cosine_threshold})"
        )

    log_level = os.environ.get("LOG_LEVEL", _DEFAULT_LOG_LEVEL)

    rate_limit_raw = os.environ.get("RATE_LIMIT_PER_MINUTE")
    if rate_limit_raw is None:
        rate_limit_per_minute = _DEFAULT_RATE_LIMIT_PER_MINUTE
    else:
        try:
            rate_limit_per_minute = int(rate_limit_raw)
        except ValueError as exc:
            raise ValueError(f"RATE_LIMIT_PER_MINUTE must be an int, got {rate_limit_raw!r}") from exc
    # A limiter built with max_requests <= 0 rejects every request, so the
    # container would boot, report healthy, and answer 429 to everyone.
    if rate_limit_per_minute <= 0:
        raise ValueError(
            f"RATE_LIMIT_PER_MINUTE must be a positive int, got {rate_limit_per_minute!r}"
        )

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

    index_profile_raw = os.environ.get("INDEX_PROFILE", _DEFAULT_INDEX_PROFILE)
    if index_profile_raw not in _VALID_INDEX_PROFILES:
        raise ValueError(
            f"INDEX_PROFILE must be one of {_VALID_INDEX_PROFILES}, got {index_profile_raw!r}"
        )
    index_profile: IndexProfileName = index_profile_raw

    deployed_sha = os.environ.get("DEPLOYED_SHA") or None
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None

    return Settings(
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        llm_provider=llm_provider,
        refusal_cosine_threshold=refusal_cosine_threshold,
        refusal_policy=refusal_policy,
        refusal_review_floor=refusal_review_floor,
        log_level=log_level,
        rate_limit_per_minute=rate_limit_per_minute,
        api_key=api_key,
        cors_allow_origins=cors_allow_origins,
        chroma_path=chroma_path,
        bm25_path=bm25_path,
        index_profile=index_profile,
        deployed_sha=deployed_sha,
        otlp_endpoint=otlp_endpoint,
    )
