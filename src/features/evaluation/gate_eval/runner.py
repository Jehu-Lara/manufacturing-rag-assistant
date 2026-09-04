from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional, cast

from src.adapters.secondary.llm.groq_openai_client import TraceHook
from src.core.config import RefusalPolicyName, Settings
from src.domain.models import Language, RetrievalResult
from src.domain.policies import top1_semantic_score_from_results
from src.domain.ports import LLMClientPort, RetrieverPort
from src.features.evaluation.gate_eval.models import (
    _POLICIES,
    PINNED_REVIEW_FLOOR,
    PINNED_THRESHOLD,
    QuestionOutcome,
    ReplayRetriever,
    RetrievalSnapshot,
    TraceCollector,
    WithinRepeatCache,
    _band,
)
from src.features.query.use_cases import QueryUseCase
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K


def capture_snapshots(
    retriever: RetrieverPort, questions: list[dict[str, Any]]
) -> tuple[list[RetrievalSnapshot], ReplayRetriever, list[float], dict[str, str]]:
    """Also returns per-retrieve latencies (so the report can state a real
    retrieval p50/p95 next to the replayed generation latency) and a
    chunk_id -> chunk_text map for the blind checklist."""
    snapshots: list[RetrievalSnapshot] = []
    by_question: dict[str, list[RetrievalResult]] = {}
    latencies_ms: list[float] = []
    chunk_text: dict[str, str] = {}
    for question in questions:
        text = question["question"]
        start = time.monotonic()
        results = retriever.retrieve(text, k=SEMANTIC_EXTRACTION_K)
        latencies_ms.append((time.monotonic() - start) * 1000)
        by_question[text] = results
        for result in results:
            chunk_text.setdefault(result.chunk_id, str(result.metadata.get("chunk_text", "")))
        score = top1_semantic_score_from_results(results)
        snapshots.append(
            RetrievalSnapshot(
                question_id=str(question["id"]),
                question=text,
                language=str(question["language"]),
                answerable=bool(question["answerable"]),
                chunk_ids=[r.chunk_id for r in results[:5]],
                top1_semantic=score,
                gate_band=_band(score),
            )
        )
    return snapshots, ReplayRetriever(by_question), latencies_ms, chunk_text


def _use_case(
    policy: RefusalPolicyName, retriever: RetrieverPort, llm: LLMClientPort, settings: Settings
) -> QueryUseCase:
    pinned = settings.model_copy(
        update={
            "refusal_policy": policy,
            "refusal_cosine_threshold": PINNED_THRESHOLD,
            "refusal_review_floor": PINNED_REVIEW_FLOOR,
        }
    )
    return QueryUseCase(retriever, llm, pinned)


def _lang(value: str) -> Language:
    if value not in ("en", "es"):
        raise ValueError(f"unsupported language {value!r}")
    return cast("Language", value)


async def _run_question(
    use_case: QueryUseCase,
    cache: WithinRepeatCache,
    trace: TraceCollector,
    *,
    repeat: int,
    policy: str,
    question: dict[str, Any],
) -> QuestionOutcome:
    trace.reset()
    logical_before, forwarded_before = cache.logical_calls, cache.forwarded_calls
    expected = [c for c in question.get("expected_chunk_ids", []) if c]
    start = time.monotonic()
    error_type: Optional[str] = None
    answer = None
    try:
        answer = await use_case.answer_question(question["question"], _lang(question["language"]))
    except Exception as exc:  # noqa: BLE001 - recorded, never aborts the matrix
        error_type = type(exc).__name__
    wall_ms = (time.monotonic() - start) * 1000

    return QuestionOutcome(
        repeat=repeat,
        policy=policy,
        question_id=str(question["id"]),
        language=str(question["language"]),
        answerable=bool(question["answerable"]),
        refused=answer.refused if answer is not None else False,
        status=answer.status if answer is not None else "error",
        gate_band=answer.gate_band if answer is not None else "n/a",
        decision_reason=answer.decision_reason if answer is not None else "runner_exception",
        confidence=answer.confidence if answer is not None else None,
        citation_count=len(answer.citations) if answer is not None else 0,
        cited_chunk_ids=[c.chunk_id for c in answer.citations] if answer is not None else [],
        expected_chunk_ids=expected,
        answer_text="" if answer is None or answer.refused else answer.answer,
        question_wall_ms=wall_ms,
        llm_latencies_ms=trace.llm_latencies_ms,
        logical_calls=cache.logical_calls - logical_before,
        forwarded_calls=cache.forwarded_calls - forwarded_before,
        physical_attempts=trace.physical_attempts,
        physical_success=trace.physical_success,
        physical_failed=trace.physical_failed,
        rate_limited=trace.rate_limited,
        repaired=trace.repaired,
        schema_fallbacks=trace.schema_fallbacks,
        provider_fallbacks=trace.provider_fallbacks,
        total_tokens=trace.total_tokens,
        error_type=error_type,
    )


def run_matrix(
    questions: list[dict[str, Any]],
    replay: ReplayRetriever,
    settings: Settings,
    llm_factory: Callable[[TraceHook], LLMClientPort],
    *,
    repeats: int,
) -> list[QuestionOutcome]:
    """One owning event loop per call: each per-repeat LLM client is created,
    used and closed inside this same loop. A cached httpx-based SDK client
    must never cross `asyncio.run` boundaries — its connections stay bound
    to the loop that opened them."""

    async def _run_all() -> list[QuestionOutcome]:
        outcomes: list[QuestionOutcome] = []
        for repeat in range(1, repeats + 1):
            trace = TraceCollector()
            llm = llm_factory(trace)
            try:
                cache = WithinRepeatCache(llm)
                for policy in _POLICIES:  # binary first so the confident call is cached
                    use_case = _use_case(policy, replay, cache, settings)
                    for question in questions:
                        outcomes.append(
                            await _run_question(
                                use_case, cache, trace, repeat=repeat, policy=policy, question=question
                            )
                        )
            finally:
                maybe_close = getattr(llm, "aclose", None)
                if callable(maybe_close):
                    await maybe_close()
        return outcomes

    return asyncio.run(_run_all())
