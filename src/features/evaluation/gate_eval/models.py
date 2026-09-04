from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, cast

from src.adapters.secondary.llm.groq_openai_client import LlmTraceEvent
from src.core.config import (
    DEFAULT_REFUSAL_COSINE_THRESHOLD,
    DEFAULT_REFUSAL_REVIEW_FLOOR,
    RefusalPolicyName,
)
from src.domain.models import ExpansionMode, IndexProfile, RetrievalResult
from src.domain.ports import LLMClientPort

# Every axis this runner may touch is pinned here, not read from the environment
# - a paid causal comparison must not silently measure a different profile,
# expansion mode, floor, threshold or provider than the one under review.
PINNED_INDEX_PROFILE: IndexProfile = "contextual-v1"
PINNED_EXPANSION_MODE: ExpansionMode = "off"
PINNED_REVIEW_FLOOR = DEFAULT_REFUSAL_REVIEW_FLOOR
PINNED_THRESHOLD = DEFAULT_REFUSAL_COSINE_THRESHOLD

FULL_REPEATS = 3
CANARY_REPEATS = 3
CANARY_MUST_ANSWER = ("r001", "r002")
CANARY_MUST_REFUSE = ("r018", "r019", "r020")

_POLICIES: tuple[RefusalPolicyName, ...] = ("binary", "grounded_review")

_PHYSICAL_EVENTS = ("physical_request", "physical_failed")


def _schema_key(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


class TraceCollector:
    """Buckets LlmTraceEvents for the currently-running question. Counts every
    physical provider round trip - `physical_attempts` includes 429s, schema
    fallbacks and network failures, not only the calls that returned JSON."""

    def __init__(self) -> None:
        self.events: list[LlmTraceEvent] = []

    def __call__(self, event: LlmTraceEvent) -> None:
        self.events.append(event)

    def reset(self) -> None:
        self.events = []

    def _count(self, *names: str) -> int:
        return sum(1 for e in self.events if e.event in names)

    @property
    def physical_attempts(self) -> int:
        return self._count("physical_attempt")

    @property
    def physical_success(self) -> int:
        return self._count("physical_request")

    @property
    def physical_failed(self) -> int:
        return self._count("physical_failed")

    @property
    def rate_limited(self) -> int:
        return self._count("rate_limited", "rate_limit_exhausted")

    @property
    def repaired(self) -> int:
        return self._count("repair_triggered")

    @property
    def schema_fallbacks(self) -> int:
        return self._count("schema_fallback")

    @property
    def provider_fallbacks(self) -> int:
        return self._count("provider_fallback")

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens or 0 for e in self.events if e.event == "physical_request")

    @property
    def llm_latencies_ms(self) -> list[float]:
        return [e.latency_ms for e in self.events if e.event in _PHYSICAL_EVENTS and e.latency_ms is not None]


class WithinRepeatCache:
    """Wraps one real LLM client for the length of ONE repeat. A byte-identical
    (system_prompt, user_prompt, schema) triple is answered once and reused -
    this is what lets the confident band's identical call be shared between the
    binary and grounded_review runs of the same repeat, and never across
    repeats (a fresh cache per repeat)."""

    def __init__(self, inner: LLMClientPort) -> None:
        self._inner = inner
        self._cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.logical_calls = 0
        self.forwarded_calls = 0

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.logical_calls += 1
        key = (system_prompt, user_prompt, _schema_key(schema))
        if key in self._cache:
            return cast("dict[str, Any]", json.loads(json.dumps(self._cache[key])))
        self.forwarded_calls += 1
        result = await self._inner.generate_structured(system_prompt, user_prompt, schema)
        self._cache[key] = cast("dict[str, Any]", json.loads(json.dumps(result)))
        return result


class ReplayRetriever:
    """Serves the frozen retrieval snapshot: identical results to every
    QueryUseCase, so binary vs grounded_review differ ONLY in policy."""

    def __init__(self, by_question: dict[str, list[RetrievalResult]]) -> None:
        self._by_question = by_question

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        if query_text not in self._by_question:
            raise KeyError(f"no retrieval snapshot captured for {query_text!r}")
        return self._by_question[query_text][:k]


@dataclass(frozen=True)
class RetrievalSnapshot:
    question_id: str
    question: str
    language: str
    answerable: bool
    chunk_ids: list[str]
    top1_semantic: Optional[float]
    gate_band: str


@dataclass(frozen=True)
class QuestionOutcome:
    repeat: int
    policy: str
    question_id: str
    language: str
    answerable: bool
    refused: bool
    status: str
    gate_band: str
    decision_reason: str
    confidence: Optional[float]
    citation_count: int
    cited_chunk_ids: list[str]
    expected_chunk_ids: list[str]
    answer_text: str
    question_wall_ms: float
    llm_latencies_ms: list[float]
    logical_calls: int
    forwarded_calls: int
    physical_attempts: int
    physical_success: int
    physical_failed: int
    rate_limited: int
    repaired: int
    schema_fallbacks: int
    provider_fallbacks: int
    total_tokens: int
    error_type: Optional[str]

    @property
    def is_unsafe_unanswerable(self) -> bool:
        return (not self.answerable) and (not self.refused) and self.status == "ok"

    @property
    def cites_all_expected(self) -> bool:
        return bool(self.expected_chunk_ids) and set(self.expected_chunk_ids).issubset(self.cited_chunk_ids)


def _band(score: Optional[float]) -> str:
    if score is None or score < PINNED_REVIEW_FLOOR:
        return "hard_refuse"
    if score < PINNED_THRESHOLD:
        return "grounded_review"
    return "confident"


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
