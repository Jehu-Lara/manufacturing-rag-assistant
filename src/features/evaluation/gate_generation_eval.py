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
    regression_set_integrity,
)
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.query.use_cases import QueryUseCase
from src.features.retrieval import index_manifest
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

REPORT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"

# Every axis this runner is allowed to touch is pinned here, not taken from the
# environment - a paid causal comparison must not silently measure a different
# index profile, expansion mode, floor or threshold than the one under review.
PINNED_INDEX_PROFILE: IndexProfile = "contextual-v1"
PINNED_EXPANSION_MODE: ExpansionMode = "off"
PINNED_REVIEW_FLOOR = DEFAULT_REFUSAL_REVIEW_FLOOR
PINNED_THRESHOLD = DEFAULT_REFUSAL_COSINE_THRESHOLD

FULL_REPEATS = 3
CANARY_REPEATS = 3
# The reported false-refusals (must answer) and the frozen decoy controls (must
# refuse). Pulled from regression_queries.json, run as their own 3x loop.
CANARY_MUST_ANSWER = ("r001", "r002")
CANARY_MUST_REFUSE = ("r018", "r019", "r020")

_POLICIES: tuple[RefusalPolicyName, ...] = ("binary", "grounded_review")


def _schema_key(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


class TraceCollector:
    """Buckets LlmTraceEvents for the currently-running question. `physical`
    counts real provider round trips (initial + repair + schema fallback);
    events carry no prompt/answer/key text (enforced by LlmTraceEvent)."""

    def __init__(self) -> None:
        self.events: list[LlmTraceEvent] = []

    def __call__(self, event: LlmTraceEvent) -> None:
        self.events.append(event)

    def reset(self) -> None:
        self.events = []

    @property
    def physical(self) -> int:
        return sum(1 for e in self.events if e.event == "physical_request")

    @property
    def rate_limited(self) -> int:
        return sum(1 for e in self.events if e.event in ("rate_limited", "rate_limit_exhausted"))

    @property
    def repaired(self) -> int:
        return sum(1 for e in self.events if e.event == "repair_triggered")

    @property
    def schema_fallbacks(self) -> int:
        return sum(1 for e in self.events if e.event == "schema_fallback")

    @property
    def provider_fallbacks(self) -> int:
        # allow_provider_fallback is False for this runner, so any of these is a bug.
        return sum(1 for e in self.events if e.event == "provider_call_failed")

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens or 0 for e in self.events if e.event == "physical_request")


class WithinRepeatCache:
    """Wraps one real LLM client for the length of ONE repeat. A byte-identical
    (system_prompt, user_prompt, schema) triple is answered once and reused -
    this is what lets the confident band's identical call be shared between the
    binary and grounded_review runs of the same repeat WITHOUT ever reusing a
    response across repeats (a fresh cache per repeat)."""

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
    latency_ms: float
    logical_calls: int
    forwarded_calls: int
    physical_requests: int
    rate_limited: int
    repaired: int
    schema_fallbacks: int
    provider_fallbacks: int
    error_type: Optional[str]


def _band(score: Optional[float]) -> str:
    if score is None or score < PINNED_REVIEW_FLOOR:
        return "hard_refuse"
    if score < PINNED_THRESHOLD:
        return "grounded_review"
    return "confident"


def capture_snapshots(
    retriever: RetrieverPort, questions: list[dict[str, Any]]
) -> tuple[list[RetrievalSnapshot], ReplayRetriever]:
    snapshots: list[RetrievalSnapshot] = []
    by_question: dict[str, list[RetrievalResult]] = {}
    for question in questions:
        text = question["question"]
        results = retriever.retrieve(text, k=SEMANTIC_EXTRACTION_K)
        by_question[text] = results
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
    return snapshots, ReplayRetriever(by_question)


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
    start = time.monotonic()
    error_type: Optional[str] = None
    try:
        answer = await use_case.answer_question(question["question"], _lang(question["language"]))
    except Exception as exc:  # noqa: BLE001 - recorded, never aborts the matrix
        latency_ms = (time.monotonic() - start) * 1000
        return QuestionOutcome(
            repeat=repeat,
            policy=policy,
            question_id=str(question["id"]),
            language=str(question["language"]),
            answerable=bool(question["answerable"]),
            refused=False,
            status="error",
            gate_band="n/a",
            decision_reason="runner_exception",
            confidence=None,
            citation_count=0,
            latency_ms=latency_ms,
            logical_calls=cache.logical_calls - logical_before,
            forwarded_calls=cache.forwarded_calls - forwarded_before,
            physical_requests=trace.physical,
            rate_limited=trace.rate_limited,
            repaired=trace.repaired,
            schema_fallbacks=trace.schema_fallbacks,
            provider_fallbacks=trace.provider_fallbacks,
            error_type=type(exc).__name__,
        )
    latency_ms = (time.monotonic() - start) * 1000
    return QuestionOutcome(
        repeat=repeat,
        policy=policy,
        question_id=str(question["id"]),
        language=str(question["language"]),
        answerable=bool(question["answerable"]),
        refused=answer.refused,
        status=answer.status,
        gate_band=answer.gate_band,
        decision_reason=answer.decision_reason,
        confidence=answer.confidence,
        citation_count=len(answer.citations),
        latency_ms=latency_ms,
        logical_calls=cache.logical_calls - logical_before,
        forwarded_calls=cache.forwarded_calls - forwarded_before,
        physical_requests=trace.physical,
        rate_limited=trace.rate_limited,
        repaired=trace.repaired,
        schema_fallbacks=trace.schema_fallbacks,
        provider_fallbacks=trace.provider_fallbacks,
        error_type=error_type,
    )


def _lang(value: str) -> Language:
    if value not in ("en", "es"):
        raise ValueError(f"unsupported language {value!r}")
    return value  # type: ignore[return-value]


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
        # binary first so the confident-band call is cached before grounded reuses it.
        for policy in _POLICIES:
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


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def _rate(rows: list[QuestionOutcome], *, answerable: bool, predicate: Callable[[QuestionOutcome], bool]) -> float:
    subset = [r for r in rows if r.answerable is answerable]
    if not subset:
        return 0.0
    return sum(1 for r in subset if predicate(r)) / len(subset)


def evaluate_gates(
    holdout: list[QuestionOutcome], canary: list[QuestionOutcome]
) -> list[GateResult]:
    gates: list[GateResult] = []

    errors = [r for r in holdout + canary if r.status == "error" or r.error_type]
    gates.append(GateResult("no_errors", not errors, f"{len(errors)} error outcome(s)"))

    fallbacks = [r for r in holdout + canary if r.provider_fallbacks or r.schema_fallbacks]
    gates.append(
        GateResult("no_provider_or_schema_fallback", not fallbacks, f"{len(fallbacks)} fallback outcome(s)")
    )

    by_policy = {p: [r for r in holdout if r.policy == p] for p in _POLICIES}
    for lang in ("global", "en", "es"):
        def _f(r: QuestionOutcome, lang: str = lang) -> bool:
            return lang == "global" or r.language == lang

        binary_correct = _rate(
            [r for r in by_policy["binary"] if _f(r)], answerable=False, predicate=lambda r: r.refused
        )
        grounded_correct = _rate(
            [r for r in by_policy["grounded_review"] if _f(r)], answerable=False, predicate=lambda r: r.refused
        )
        gates.append(
            GateResult(
                f"correct_refusal_not_worse[{lang}]",
                grounded_correct >= binary_correct - 1e-9,
                f"binary={binary_correct:.3f} grounded={grounded_correct:.3f}",
            )
        )
        binary_false = _rate(
            [r for r in by_policy["binary"] if _f(r)], answerable=True, predicate=lambda r: r.refused
        )
        grounded_false = _rate(
            [r for r in by_policy["grounded_review"] if _f(r)], answerable=True, predicate=lambda r: r.refused
        )
        worse = grounded_false > binary_false + 1e-9
        gates.append(
            GateResult(
                f"false_refusal_not_worse[{lang}]",
                not worse,
                f"binary={binary_false:.3f} grounded={grounded_false:.3f}",
            )
        )
    # global false-refusal must strictly improve
    gb_false = _rate(by_policy["binary"], answerable=True, predicate=lambda r: r.refused)
    gg_false = _rate(by_policy["grounded_review"], answerable=True, predicate=lambda r: r.refused)
    gates.append(
        GateResult(
            "false_refusal_improves_global",
            gg_false < gb_false - 1e-9,
            f"binary={gb_false:.3f} grounded={gg_false:.3f}",
        )
    )

    grounded_canary = [r for r in canary if r.policy == "grounded_review"]
    for qid in CANARY_MUST_ANSWER:
        hits = [r for r in grounded_canary if r.question_id == qid]
        ok = len(hits) == CANARY_REPEATS and all(not r.refused and r.status == "ok" for r in hits)
        answered = sum(not r.refused for r in hits)
        gates.append(GateResult(f"canary_answers[{qid}]", ok, f"{answered}/{len(hits)} answered"))
    for qid in CANARY_MUST_REFUSE:
        hits = [r for r in grounded_canary if r.question_id == qid]
        ok = len(hits) == CANARY_REPEATS and all(r.refused for r in hits)
        refused = sum(r.refused for r in hits)
        gates.append(GateResult(f"canary_refuses[{qid}]", ok, f"{refused}/{len(hits)} refused"))

    gates.append(
        GateResult(
            "citation_faithfulness_human_review",
            False,
            "PENDING - grade the blind checklist by hand; conditional >= 0.90 required",
        )
    )
    return gates


def _latency_percentiles(rows: list[QuestionOutcome]) -> dict[str, float]:
    values = sorted(r.latency_ms for r in rows)
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": statistics.median(values),
        "p95": values[min(len(values) - 1, max(0, round(0.95 * len(values)) - 1))],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _blind_checklist(path: Path, snapshots: list[RetrievalSnapshot], holdout: list[QuestionOutcome]) -> None:
    answered = {
        (r.question_id, r.policy): r
        for r in holdout
        if r.answerable and not r.refused and r.status == "ok"
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question_id",
                "policy",
                "language",
                "cited_chunk_ids",
                "citation_accuracy_pass",
                "faithfulness_pass",
                "notes",
            ]
        )
        snap_by_id = {s.question_id: s for s in snapshots}
        for (qid, policy), _row in sorted(answered.items()):
            snap = snap_by_id.get(qid)
            writer.writerow([qid, policy, snap.language if snap else "", "", "", "", ""])


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
        f"**Automated gates: {'ALL PASS' if all_pass else 'NOT ALL PASS'}** "
        "(the ship decision is the owner's, after the human review line above).",
        "",
        "## Call + latency accounting",
        "",
        "| policy | logical | forwarded | physical | rate_limited | repaired | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy in _POLICIES:
        rows = [r for r in holdout if r.policy == policy]
        pct = _latency_percentiles(rows)
        lines.append(
            f"| {policy} | {sum(r.logical_calls for r in rows)} | {sum(r.forwarded_calls for r in rows)} "
            f"| {sum(r.physical_requests for r in rows)} | {sum(r.rate_limited for r in rows)} "
            f"| {sum(r.repaired for r in rows)} | {pct['p50']:.0f} | {pct['p95']:.0f} |"
        )

    lines += ["", "## Retrieval snapshot band distribution", "", "| band | count |", "|---|---|"]
    band_counts: dict[str, int] = {}
    for snap in snapshots:
        band_counts[snap.gate_band] = band_counts.get(snap.gate_band, 0) + 1
    for band, count in sorted(band_counts.items()):
        lines.append(f"| {band} | {count} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_run_dir(
    out_root: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    snapshots: list[RetrievalSnapshot],
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    gates: list[GateResult],
) -> Path:
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a prior run

    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(run_dir / "retrieval.jsonl", [asdict(s) for s in snapshots])
    _write_jsonl(run_dir / "outcomes.jsonl", [asdict(o) for o in holdout + canary])
    (run_dir / "comparison.md").write_text(
        render_comparison(manifest, snapshots, holdout, canary, gates), encoding="utf-8"
    )
    _blind_checklist(run_dir / "blind_checklist.csv", snapshots, holdout)

    checksums = "\n".join(
        f"{_sha256_file(p)}  {p.name}"
        for p in sorted(run_dir.iterdir())
        if p.name != "checksums.txt"
    )
    (run_dir / "checksums.txt").write_text(checksums + "\n", encoding="utf-8")
    return run_dir


@dataclass
class _Prereqs:
    settings: Settings
    build_commit: str


def _verify_prereqs(holdout_path: Path, regression_path: Path) -> _Prereqs:
    eval_set_integrity.verify()
    regression_set_integrity.verify(regression_path)
    gate_holdout_integrity.verify(holdout_path, regression_set_path=regression_path)
    assert_live_index_profile(PINNED_INDEX_PROFILE)
    manifest = index_manifest.read()
    if manifest.index_profile != PINNED_INDEX_PROFILE:
        raise RuntimeError(f"live index profile {manifest.index_profile!r} != pinned {PINNED_INDEX_PROFILE!r}")
    settings = load_settings()
    return _Prereqs(settings=settings, build_commit=manifest.build_commit)


def run(
    *,
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
    """`retriever`/`llm_factory`/`settings` are injectable so the whole matrix
    runs against fakes in tests. A real invocation (no injection) verifies the
    frozen datasets + live index, then does `full_repeats` paid passes over the
    holdout and `canary_repeats` over r001/r002/r018-r020."""
    if retriever is None or settings is None or build_commit is None:
        prereqs = _verify_prereqs(holdout_path, regression_path)
        settings = settings or prereqs.settings
        build_commit = build_commit or prereqs.build_commit
        retriever = retriever or build_retriever(PINNED_EXPANSION_MODE, expected_profile=PINNED_INDEX_PROFILE)

    if llm_factory is None:
        def llm_factory(hook: TraceHook) -> LLMClientPort:
            return GroqOpenAiLlmClient(allow_provider_fallback=False, trace_hook=hook)

    holdout_questions = gate_holdout_integrity.load_gate_holdout(holdout_path)["questions"]
    regression_by_id = {
        q["id"]: q for q in regression_set_integrity.load_regression_set(regression_path)["queries"]
    }
    canary_questions = [
        {
            "id": qid,
            "question": regression_by_id[qid]["query"],
            "language": regression_by_id[qid]["language"],
            "answerable": qid in CANARY_MUST_ANSWER,
        }
        for qid in (*CANARY_MUST_ANSWER, *CANARY_MUST_REFUSE)
    ]

    snapshots, holdout_replay = capture_snapshots(retriever, holdout_questions)
    canary_snaps, canary_replay = capture_snapshots(retriever, canary_questions)

    holdout_outcomes = run_matrix(
        holdout_questions, holdout_replay, settings, llm_factory, repeats=full_repeats
    )
    canary_outcomes = run_matrix(
        canary_questions, canary_replay, settings, llm_factory, repeats=canary_repeats
    )

    gates = evaluate_gates(holdout_outcomes, canary_outcomes)

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"gate_generation_eval_{stamp}"
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
        "eval_set_sha256": eval_set_integrity.compute_hash(
            eval_set_integrity.load_eval_set()["questions"]
        ),
        "regression_sha256": regression_set_integrity.compute_hash(
            regression_set_integrity.load_regression_set(regression_path)["queries"]
        ),
        "holdout_sha256": gate_holdout_integrity.load_gate_holdout(holdout_path)["sha256"],
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
    )
    print(f"Run written to: {run_dir}")
    for gate in gates:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-repeats", type=int, default=FULL_REPEATS)
    parser.add_argument("--canary-repeats", type=int, default=CANARY_REPEATS)
    args = parser.parse_args()
    run(full_repeats=args.full_repeats, canary_repeats=args.canary_repeats)


if __name__ == "__main__":
    main()
