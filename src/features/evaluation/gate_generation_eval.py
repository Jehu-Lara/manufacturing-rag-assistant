from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, cast

from src.adapters.secondary.llm.groq_openai_client import GroqOpenAiLlmClient, LlmTraceEvent, TraceHook
from src.core.config import (
    DEFAULT_REFUSAL_COSINE_THRESHOLD,
    DEFAULT_REFUSAL_REVIEW_FLOOR,
    RefusalPolicyName,
    Settings,
    load_settings,
)
from src.domain.models import ExpansionMode, IndexProfile, Language, RetrievalResult
from src.domain.policies import top1_semantic_score_from_results
from src.domain.ports import LLMClientPort, RetrieverPort
from src.features.evaluation import (
    eval_set_integrity,
    gate_holdout_integrity,
    gate_holdout_profile,
    regression_set_integrity,
)
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.query.use_cases import QueryUseCase
from src.features.retrieval import index_manifest
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

REPORT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"

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
        return self._count("provider_call_failed", "generation_exhausted")

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
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], settings: Settings
    ) -> dict[str, Any]:
        self.logical_calls += 1
        key = (system_prompt, user_prompt, _schema_key(schema))
        if key in self._cache:
            return cast("dict[str, Any]", json.loads(json.dumps(self._cache[key])))
        self.forwarded_calls += 1
        result = await self._inner.generate_structured(system_prompt, user_prompt, schema, settings)
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
    outcomes: list[QuestionOutcome] = []
    for repeat in range(1, repeats + 1):
        trace = TraceCollector()
        cache = WithinRepeatCache(llm_factory(trace))
        for policy in _POLICIES:  # binary first so the confident call is cached
            use_case = _use_case(policy, replay, cache, settings)
            for question in questions:
                outcomes.append(
                    asyncio.run(
                        _run_question(
                            use_case, cache, trace, repeat=repeat, policy=policy, question=question
                        )
                    )
                )
    return outcomes


# --------------------------------------------------------------------------- #
# Blind checklist                                                             #
# --------------------------------------------------------------------------- #

_CHECKLIST_HEADER = [
    "row_id",
    "arm",
    "repeat",
    "question_id",
    "language",
    "answerable",
    "refused",
    "answer",
    "cited_chunk_ids",
    "cited_chunk_texts",
    "expected_answer",
    "expected_chunk_ids",
    "citation_accuracy_pass",
    "faithfulness_pass",
    "safe_pass",
    "notes",
]
_EDITABLE_COLUMNS = ("citation_accuracy_pass", "faithfulness_pass", "safe_pass", "notes")
_VERDICT_COLUMNS = ("citation_accuracy_pass", "faithfulness_pass", "safe_pass")
_IMMUTABLE_COLUMNS = tuple(c for c in _CHECKLIST_HEADER if c not in _EDITABLE_COLUMNS)


def _arm_labels(run_id: str) -> dict[str, str]:
    """Deterministic but not guessable: order the two policies by a hash of
    (run_id, policy). Grader sees `arm-A` / `arm-B`, the mapping lives only in
    arm_map.sealed.json."""
    ordered = sorted(_POLICIES, key=lambda p: hashlib.sha256(f"{run_id}|{p}".encode()).hexdigest())
    return {ordered[0]: "arm-A", ordered[1]: "arm-B"}


def _checklist_rows(
    run_id: str,
    outcomes: list[QuestionOutcome],
    chunk_text: dict[str, str],
    expected_answers: dict[str, str],
) -> list[dict[str, str]]:
    arm = _arm_labels(run_id)
    rows: list[dict[str, str]] = []
    for outcome in outcomes:
        needs_grading = (outcome.answerable and not outcome.refused and outcome.status == "ok") or (
            outcome.is_unsafe_unanswerable
        )
        if not needs_grading:
            continue
        rows.append(
            {
                "row_id": f"{arm[outcome.policy]}-r{outcome.repeat}-{outcome.question_id}",
                "arm": arm[outcome.policy],
                "repeat": str(outcome.repeat),
                "question_id": outcome.question_id,
                "language": outcome.language,
                "answerable": str(outcome.answerable),
                "refused": str(outcome.refused),
                "answer": outcome.answer_text,
                "cited_chunk_ids": ";".join(outcome.cited_chunk_ids),
                "cited_chunk_texts": " || ".join(
                    chunk_text.get(cid, "<not in snapshot>") for cid in outcome.cited_chunk_ids
                ),
                "expected_answer": expected_answers.get(outcome.question_id, ""),
                "expected_chunk_ids": ";".join(outcome.expected_chunk_ids),
                "citation_accuracy_pass": "",
                "faithfulness_pass": "",
                "safe_pass": "",
                "notes": "",
            }
        )
    return rows


def _write_checklist(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CHECKLIST_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def checklist_baseline(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """The immutable half of every checklist row, keyed by row_id. Sealed at run
    time; on import the graded CSV must reproduce it exactly - a deleted,
    added, duplicated or altered row is rejected before any gate is scored."""
    return {row["row_id"]: {col: row[col] for col in _IMMUTABLE_COLUMNS} for row in rows}


@dataclass
class HumanVerdicts:
    graded_rows: int
    citation_pass_rate: float
    faithfulness_pass_rate: float
    unsafe_unanswerable_rows: int
    unsafe_all_safe: bool


def _parse_pass(value: str) -> Optional[bool]:
    token = value.strip().lower()
    if token in ("y", "yes", "true", "1", "pass"):
        return True
    if token in ("n", "no", "false", "0", "fail"):
        return False
    return None


def _verify_against_baseline(
    rows: list[dict[str, str]], baseline: dict[str, dict[str, str]]
) -> None:
    for row in rows:
        if set(row) != set(_CHECKLIST_HEADER):
            raise ValueError(
                f"blind checklist row has unexpected columns: {sorted(set(row) ^ set(_CHECKLIST_HEADER))}"
            )
    ids = [r["row_id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"blind checklist has duplicate row_id(s): {dupes}")
    missing = sorted(set(baseline) - set(ids))
    extra = sorted(set(ids) - set(baseline))
    if missing or extra:
        raise ValueError(
            f"blind checklist row set does not match the sealed baseline - missing {missing}, extra {extra}. "
            "Grade the file the runner produced; do not add, delete or reorder rows."
        )
    for row in rows:
        expected = baseline[row["row_id"]]
        drifted = [c for c in _IMMUTABLE_COLUMNS if row[c] != expected[c]]
        if drifted:
            raise ValueError(
                f"blind checklist row {row['row_id']!r} altered immutable column(s) {drifted}; "
                "only the pass/notes columns may be edited"
            )


def import_human_verdicts(
    checklist_path: Path, baseline: dict[str, dict[str, str]]
) -> HumanVerdicts:
    with checklist_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("blind checklist is empty")
    _verify_against_baseline(rows, baseline)

    answerable_rows = [r for r in rows if r["answerable"] == "True" and r["refused"] == "False"]
    unsafe_rows = [r for r in rows if r["answerable"] == "False" and r["refused"] == "False"]
    ungraded = any(
        _parse_pass(r[col]) is None
        for r in rows
        for col in _VERDICT_COLUMNS
        if not (col == "safe_pass" and r["answerable"] == "True")
        and not (col in ("citation_accuracy_pass", "faithfulness_pass") and r["answerable"] == "False")
    )
    if ungraded:
        raise ValueError(
            "blind checklist is not fully graded - every row needs y/n in the pass columns "
            "that apply to it (citation/faithfulness for answered questions, safe for "
            "answered unanswerables)"
        )
    cite_ok = sum(1 for r in answerable_rows if _parse_pass(r["citation_accuracy_pass"]))
    faith_ok = sum(1 for r in answerable_rows if _parse_pass(r["faithfulness_pass"]))
    n = len(answerable_rows) or 1
    return HumanVerdicts(
        graded_rows=len(rows),
        citation_pass_rate=cite_ok / n,
        faithfulness_pass_rate=faith_ok / n,
        unsafe_unanswerable_rows=len(unsafe_rows),
        unsafe_all_safe=all(_parse_pass(r["safe_pass"]) for r in unsafe_rows),
    )


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def _rate(
    rows: list[QuestionOutcome], *, answerable: bool, predicate: Callable[[QuestionOutcome], bool]
) -> float:
    subset = [r for r in rows if r.answerable is answerable]
    if not subset:
        return 0.0
    return sum(1 for r in subset if predicate(r)) / len(subset)


def evaluate_gates(
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    verdicts: Optional[HumanVerdicts] = None,
) -> list[GateResult]:
    gates: list[GateResult] = []
    every = holdout + canary

    errors = [r for r in every if r.status == "error" or r.error_type]
    gates.append(GateResult("no_errors", not errors, f"{len(errors)} error outcome(s)"))

    fallbacks = [r for r in every if r.provider_fallbacks or r.schema_fallbacks]
    gates.append(
        GateResult("no_provider_or_schema_fallback", not fallbacks, f"{len(fallbacks)} fallback outcome(s)")
    )

    limited = [r for r in every if r.rate_limited]
    gates.append(
        GateResult(
            "no_rate_limiting",
            not limited,
            f"{len(limited)} outcome(s) hit a 429 - re-run on a non-rate-limited key",
        )
    )

    unsafe = [r for r in every if r.is_unsafe_unanswerable]
    if verdicts is None:
        gates.append(
            GateResult(
                "no_unsafe_unanswerable_answer",
                not unsafe,
                f"{len(unsafe)} unanswerable question(s) were answered - needs blind safe_pass review"
                if unsafe
                else "0 unanswerable questions were answered",
            )
        )
    else:
        gates.append(
            GateResult(
                "no_unsafe_unanswerable_answer",
                verdicts.unsafe_all_safe,
                f"{verdicts.unsafe_unanswerable_rows} answered-unanswerable row(s), all graded safe: "
                f"{verdicts.unsafe_all_safe}",
            )
        )

    by_policy = {p: [r for r in holdout if r.policy == p] for p in _POLICIES}
    for lang in ("global", "en", "es"):

        def _f(r: QuestionOutcome, lang: str = lang) -> bool:
            return lang == "global" or r.language == lang

        b_correct = _rate([r for r in by_policy["binary"] if _f(r)], answerable=False, predicate=lambda r: r.refused)
        g_correct = _rate(
            [r for r in by_policy["grounded_review"] if _f(r)], answerable=False, predicate=lambda r: r.refused
        )
        gates.append(
            GateResult(
                f"correct_refusal_not_worse[{lang}]",
                g_correct >= b_correct - 1e-9,
                f"binary={b_correct:.3f} grounded={g_correct:.3f}",
            )
        )
        b_false = _rate([r for r in by_policy["binary"] if _f(r)], answerable=True, predicate=lambda r: r.refused)
        g_false = _rate(
            [r for r in by_policy["grounded_review"] if _f(r)], answerable=True, predicate=lambda r: r.refused
        )
        gates.append(
            GateResult(
                f"false_refusal_not_worse[{lang}]",
                g_false <= b_false + 1e-9,
                f"binary={b_false:.3f} grounded={g_false:.3f}",
            )
        )
    gb = _rate(by_policy["binary"], answerable=True, predicate=lambda r: r.refused)
    gg = _rate(by_policy["grounded_review"], answerable=True, predicate=lambda r: r.refused)
    gates.append(
        GateResult("false_refusal_improves_global", gg < gb - 1e-9, f"binary={gb:.3f} grounded={gg:.3f}")
    )

    grounded_canary = [r for r in canary if r.policy == "grounded_review"]
    for qid in CANARY_MUST_ANSWER:
        hits = [r for r in grounded_canary if r.question_id == qid]
        answered_ok = [r for r in hits if not r.refused and r.status == "ok"]
        cited_ok = [r for r in answered_ok if r.cites_all_expected]
        ok = len(hits) == CANARY_REPEATS and len(cited_ok) == CANARY_REPEATS
        gates.append(
            GateResult(
                f"canary_answers_and_cites[{qid}]",
                ok,
                f"{len(answered_ok)}/{len(hits)} answered, {len(cited_ok)}/{len(hits)} cite expected chunk "
                "(entailment still needs blind grading)",
            )
        )
    for qid in CANARY_MUST_REFUSE:
        hits = [r for r in grounded_canary if r.question_id == qid]
        refused = sum(r.refused for r in hits)
        gates.append(
            GateResult(
                f"canary_refuses[{qid}]",
                len(hits) == CANARY_REPEATS and refused == CANARY_REPEATS,
                f"{refused}/{len(hits)} refused",
            )
        )

    if verdicts is None:
        gates.append(
            GateResult(
                "citation_faithfulness_conditional",
                False,
                "PENDING - grade blind_checklist.csv then run "
                "`gate_generation_eval --import-verdicts <run_dir>`; need both >= 0.90",
            )
        )
    else:
        ok = verdicts.citation_pass_rate >= 0.90 and verdicts.faithfulness_pass_rate >= 0.90
        gates.append(
            GateResult(
                "citation_faithfulness_conditional",
                ok,
                f"citation={verdicts.citation_pass_rate:.3f} faithfulness={verdicts.faithfulness_pass_rate:.3f} "
                f"over {verdicts.graded_rows} graded rows",
            )
        )
    return gates


# --------------------------------------------------------------------------- #
# Reporting + artifacts                                                       #
# --------------------------------------------------------------------------- #


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def render_comparison(
    manifest: dict[str, Any],
    snapshots: list[RetrievalSnapshot],
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    gates: list[GateResult],
) -> str:
    lines = [
        "# Gate Generation Eval - binary vs grounded_review",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Index: `{manifest['index_profile']}` / `{manifest['expansion_mode']}`  build `{manifest['build_commit']}`",
        f"- Floor/threshold: {manifest['review_floor']} / {manifest['threshold']}",
        f"- Provider: `{manifest['llm_provider']}` (provider fallback disabled)",
        f"- Repeats: holdout x{manifest['full_repeats']}, canary x{manifest['canary_repeats']}",
        f"- Verdicts imported: {manifest.get('verdicts_imported', False)}",
        "",
        "## Gate verdicts",
        "",
        "| gate | verdict | detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.detail} |")
    all_pass = all(g.passed for g in gates)
    lines += [
        "",
        f"**Automated gates: {'ALL PASS' if all_pass else 'NOT ALL PASS'}** - the ship decision, the "
        "default flip and ADR-009 Accepted are the owner's, not this runner's.",
        "",
        "## Call + cost accounting (holdout)",
        "",
        "| policy | logical | forwarded | physical attempts | failed | rate-limited | repaired | tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy in _POLICIES:
        rows = [r for r in holdout if r.policy == policy]
        lines.append(
            f"| {policy} | {sum(r.logical_calls for r in rows)} | {sum(r.forwarded_calls for r in rows)} "
            f"| {sum(r.physical_attempts for r in rows)} | {sum(r.physical_failed for r in rows)} "
            f"| {sum(r.rate_limited for r in rows)} | {sum(r.repaired for r in rows)} "
            f"| {sum(r.total_tokens for r in rows)} |"
        )
    gen = _percentiles([ms for r in holdout for ms in r.llm_latencies_ms])
    retr = _percentiles(manifest.get("retrieval_latencies_ms", []))
    lines += [
        "",
        "## Latency (modelled, not question wall-clock)",
        "",
        f"- Generation, per physical LLM call: p50 {gen['p50']:.0f} ms, p95 {gen['p95']:.0f} ms",
        f"- Retrieval, per query on the live index: p50 {retr['p50']:.0f} ms, p95 {retr['p95']:.0f} ms",
        "- A production grey-band request ~= one retrieval + one generation. Confident-band reuse "
        "and replayed retrieval are excluded from these figures by construction.",
        "",
        "## Retrieval snapshot band distribution",
        "",
        "| band | count |",
        "|---|---|",
    ]
    band_counts: dict[str, int] = {}
    for snap in snapshots:
        band_counts[snap.gate_band] = band_counts.get(snap.gate_band, 0) + 1
    for band, count in sorted(band_counts.items()):
        lines.append(f"| {band} | {count} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _finalize_dir(partial: Path, final: Path) -> None:
    checksums = "\n".join(
        f"{_sha256_file(p)}  {p.name}" for p in sorted(partial.iterdir()) if p.name != "checksums.txt"
    )
    (partial / "checksums.txt").write_text(checksums + "\n", encoding="utf-8")
    partial.rename(final)


def write_run_dir(
    out_root: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    snapshots: list[RetrievalSnapshot],
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    gates: list[GateResult],
    checklist_rows: list[dict[str, str]],
    arm_map: dict[str, str],
) -> Path:
    final = out_root / run_id
    if final.exists():
        raise FileExistsError(f"{final} already exists - runs are write-once, never overwritten")
    partial = out_root / f"{run_id}.partial"
    if partial.exists():
        raise FileExistsError(f"{partial} left by an aborted run - inspect and remove it by hand")
    partial.mkdir(parents=True, exist_ok=False)

    (partial / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(partial / "retrieval.jsonl", [asdict(s) for s in snapshots])
    _write_jsonl(partial / "outcomes.jsonl", [asdict(o) for o in holdout + canary])
    (partial / "comparison.md").write_text(
        render_comparison(manifest, snapshots, holdout, canary, gates), encoding="utf-8"
    )
    _write_checklist(partial / "blind_checklist.csv", checklist_rows)
    (partial / "blind_checklist.baseline.json").write_text(
        json.dumps(checklist_baseline(checklist_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (partial / "arm_map.sealed.json").write_text(
        json.dumps(arm_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _finalize_dir(partial, final)
    return final


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class _Prereqs:
    settings: Settings
    build_commit: str


def _verify_prereqs(holdout_path: Path, regression_path: Path, provider: str) -> _Prereqs:
    eval_set_integrity.verify()
    regression_set_integrity.verify(regression_path)
    gate_holdout_integrity.verify(holdout_path, regression_set_path=regression_path)
    assert_live_index_profile(PINNED_INDEX_PROFILE)
    manifest = index_manifest.read()
    if manifest.index_profile != PINNED_INDEX_PROFILE:
        raise RuntimeError(f"live index profile {manifest.index_profile!r} != pinned {PINNED_INDEX_PROFILE!r}")
    settings = load_settings()
    if settings.llm_provider != provider:
        raise RuntimeError(
            f"--provider {provider!r} does not match LLM_PROVIDER={settings.llm_provider!r}; "
            "set them consistently so the run manifest is unambiguous"
        )
    return _Prereqs(settings=settings, build_commit=manifest.build_commit)


def _assert_profile_coverage(snapshots: list[RetrievalSnapshot]) -> None:
    """The paid run is only meaningful if the holdout actually exercises the
    grey band (same check as gate_holdout_profile, enforced here so the runner
    cannot skip it)."""
    cells: dict[tuple[str, bool], int] = {}
    for snap in snapshots:
        if snap.gate_band == "grounded_review":
            cells[(snap.language, snap.answerable)] = cells.get((snap.language, snap.answerable), 0) + 1
    short = [
        f"{lang}/{'answerable' if ans else 'unanswerable'}={cells.get((lang, ans), 0)}"
        for lang in ("en", "es")
        for ans in (True, False)
        if cells.get((lang, ans), 0) < gate_holdout_profile.MIN_GREY_PER_CELL
    ]
    if short:
        raise RuntimeError(
            "holdout does not put >= "
            f"{gate_holdout_profile.MIN_GREY_PER_CELL} questions in the grounded-review band per cell "
            f"({', '.join(short)}); run gate_holdout_profile and author another draft before spending"
        )


def _canary_questions(regression_path: Path) -> list[dict[str, Any]]:
    by_id = {q["id"]: q for q in regression_set_integrity.load_regression_set(regression_path)["queries"]}
    out: list[dict[str, Any]] = []
    for qid in (*CANARY_MUST_ANSWER, *CANARY_MUST_REFUSE):
        row = by_id[qid]
        expected = row.get("expected_chunk_id")
        out.append(
            {
                "id": qid,
                "question": row["query"],
                "language": row["language"],
                "answerable": qid in CANARY_MUST_ANSWER,
                "expected_chunk_ids": [expected] if expected else [],
            }
        )
    return out


def run(
    *,
    provider: str = "groq",
    holdout_path: Path = gate_holdout_integrity.GATE_HOLDOUT_FILE,
    regression_path: Path = regression_set_integrity.REGRESSION_SET_FILE,
    out_root: Path = REPORT_ROOT,
    retriever: Optional[RetrieverPort] = None,
    llm_factory: Optional[Callable[[TraceHook], LLMClientPort]] = None,
    settings: Optional[Settings] = None,
    build_commit: Optional[str] = None,
    full_repeats: int = FULL_REPEATS,
    canary_repeats: int = CANARY_REPEATS,
    now: Optional[datetime] = None,
) -> Path:
    """`retriever`/`llm_factory`/`settings`/`build_commit` are injectable so the
    whole matrix runs against fakes in tests. A real invocation verifies the
    frozen datasets + live index + provider match, enforces grey-band coverage,
    then does `full_repeats` paid holdout passes and `canary_repeats` canary
    passes."""
    injected = retriever is not None and settings is not None and build_commit is not None
    if not injected:
        prereqs = _verify_prereqs(holdout_path, regression_path, provider)
        settings = settings or prereqs.settings
        build_commit = build_commit or prereqs.build_commit
        retriever = retriever or build_retriever(
            PINNED_EXPANSION_MODE, expected_profile=PINNED_INDEX_PROFILE
        )
    assert retriever is not None and settings is not None and build_commit is not None

    if llm_factory is None:

        def llm_factory(hook: TraceHook) -> LLMClientPort:
            return GroqOpenAiLlmClient(allow_provider_fallback=False, trace_hook=hook)

    holdout_data = gate_holdout_integrity.load_gate_holdout(holdout_path)
    holdout_questions = holdout_data["questions"]
    expected_answers = {
        str(q["id"]): str(q.get("expected_answer", ""))
        for q in holdout_questions
        if q.get("answerable")
    }
    canary_questions = _canary_questions(regression_path)

    snapshots, holdout_replay, retr_latencies, chunk_text = capture_snapshots(retriever, holdout_questions)
    canary_snaps, canary_replay, canary_latencies, canary_text = capture_snapshots(retriever, canary_questions)
    chunk_text.update(canary_text)
    if not injected:
        _assert_profile_coverage(snapshots)

    holdout_outcomes = run_matrix(holdout_questions, holdout_replay, settings, llm_factory, repeats=full_repeats)
    canary_outcomes = run_matrix(canary_questions, canary_replay, settings, llm_factory, repeats=canary_repeats)

    gates = evaluate_gates(holdout_outcomes, canary_outcomes)

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"gate_generation_eval_{stamp}"
    arm_map = {label: policy for policy, label in _arm_labels(run_id).items()}
    checklist_rows = _checklist_rows(
        run_id, holdout_outcomes + canary_outcomes, chunk_text, expected_answers
    )
    manifest = {
        "run_id": run_id,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "index_profile": PINNED_INDEX_PROFILE,
        "expansion_mode": PINNED_EXPANSION_MODE,
        "review_floor": PINNED_REVIEW_FLOOR,
        "threshold": PINNED_THRESHOLD,
        "llm_provider": settings.llm_provider,
        "build_commit": build_commit,
        "full_repeats": full_repeats,
        "canary_repeats": canary_repeats,
        "policies": list(_POLICIES),
        "verdicts_imported": False,
        "retrieval_latencies_ms": retr_latencies + canary_latencies,
        "eval_set_sha256": eval_set_integrity.compute_hash(eval_set_integrity.load_eval_set()["questions"]),
        "regression_sha256": regression_set_integrity.compute_hash(
            regression_set_integrity.load_regression_set(regression_path)["queries"]
        ),
        "holdout_sha256": holdout_data["sha256"],
        "holdout_question_count": len(holdout_questions),
    }
    run_dir = write_run_dir(
        out_root,
        run_id=run_id,
        manifest=manifest,
        snapshots=[*snapshots, *canary_snaps],
        holdout=holdout_outcomes,
        canary=canary_outcomes,
        gates=gates,
        checklist_rows=checklist_rows,
        arm_map=arm_map,
    )
    print(f"Run written to: {run_dir}")
    for gate in gates:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    return run_dir


def import_verdicts_into_run(run_dir: Path) -> Path:
    """Re-grade the gates that need the human blind review and rewrite
    comparison.md in place (checksums.txt regenerated). The raw run files are
    not touched."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    outcomes_raw = [
        json.loads(line) for line in (run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    outcomes = [QuestionOutcome(**{k: v for k, v in row.items()}) for row in outcomes_raw]
    holdout = [o for o in outcomes if o.question_id not in (*CANARY_MUST_ANSWER, *CANARY_MUST_REFUSE)]
    canary = [o for o in outcomes if o.question_id in (*CANARY_MUST_ANSWER, *CANARY_MUST_REFUSE)]
    snapshots = [
        RetrievalSnapshot(**json.loads(line))
        for line in (run_dir / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    baseline = json.loads((run_dir / "blind_checklist.baseline.json").read_text(encoding="utf-8"))
    verdicts = import_human_verdicts(run_dir / "blind_checklist.csv", baseline)

    runner_unsafe = sum(1 for o in outcomes if o.is_unsafe_unanswerable)
    if verdicts.unsafe_unanswerable_rows != runner_unsafe:
        raise ValueError(
            f"blind checklist has {verdicts.unsafe_unanswerable_rows} answered-unanswerable row(s) but "
            f"outcomes.jsonl has {runner_unsafe} - the checklist was tampered with"
        )
    gates = evaluate_gates(holdout, canary, verdicts)
    manifest["verdicts_imported"] = True
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "human_verdicts.json").write_text(
        json.dumps(asdict(verdicts), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "comparison.md").write_text(
        render_comparison(manifest, snapshots, holdout, canary, gates), encoding="utf-8"
    )
    checksums = "\n".join(
        f"{_sha256_file(p)}  {p.name}" for p in sorted(run_dir.iterdir()) if p.name != "checksums.txt"
    )
    (run_dir / "checksums.txt").write_text(checksums + "\n", encoding="utf-8")
    for gate in gates:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    return run_dir


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "openai"], default=None)
    parser.add_argument("--full-repeats", type=int, default=FULL_REPEATS)
    parser.add_argument("--canary-repeats", type=int, default=CANARY_REPEATS)
    parser.add_argument("--import-verdicts", type=Path, default=None, metavar="RUN_DIR")
    args = parser.parse_args(argv)

    if args.import_verdicts is not None:
        import_verdicts_into_run(args.import_verdicts)
        return
    if args.provider is None:
        parser.error("--provider {groq|openai} is required for a paid run")
    run(provider=args.provider, full_repeats=args.full_repeats, canary_repeats=args.canary_repeats)


if __name__ == "__main__":
    main()
