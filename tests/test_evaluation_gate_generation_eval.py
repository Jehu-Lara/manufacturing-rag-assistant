from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.adapters.secondary.llm.groq_openai_client import LlmTraceEvent
from src.core.config import Settings
from src.domain.models import RetrievalResult
from src.features.evaluation import gate_generation_eval as gge

_GREY_TEXT = (
    "Net positive suction head available (NPSHA) is a property of the system; net positive "
    "suction head required (NPSHR) is a property of the pump and is set by the manufacturer."
)


def _settings() -> Settings:
    return Settings(
        groq_api_key="k",
        openai_api_key="k",
        llm_provider="groq",
        refusal_cosine_threshold=gge.PINNED_THRESHOLD,
        refusal_review_floor=gge.PINNED_REVIEW_FLOOR,
        log_level="INFO",
    )


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=1.0,
        semantic_rank=1,
        semantic_score=score,
        bm25_rank=1,
        bm25_score=1.0,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "d",
            "document_title": "T",
            "section_heading": "S",
            "revision": "A",
            "chunk_text": _GREY_TEXT,
        },
    )


class _ScriptedRetriever:
    def __init__(self, score_by_text: dict[str, float]) -> None:
        self._score_by_text = score_by_text

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        return [_result("chunk-1", self._score_by_text[query_text])]


def _fake_llm_factory(emit_physical: bool = True):
    def factory(hook):
        class _Fake:
            async def generate_structured(self, system_prompt, user_prompt, schema, settings):
                if emit_physical:
                    hook(
                        LlmTraceEvent(
                            event="physical_request", provider="groq", phase="initial", total_tokens=7
                        )
                    )
                if "evidence" in schema["properties"]:
                    quote = "net positive suction head required (NPSHR) is a property of the pump"
                    return {
                        "answer": "A grounded answer.",
                        "evidence": [{"chunk_id": "chunk-1", "supporting_quote": quote}],
                        "refused": False,
                    }
                return {"answer": "A confident answer.", "citations": [{"chunk_id": "chunk-1"}], "refused": False}

        return _Fake()

    return factory


def _holdout_questions() -> list[dict]:
    questions = []
    for n in range(3):
        for lang in ("en", "es"):
            questions.append(
                {
                    "id": f"h{n}{lang}",
                    "pair_id": f"h{n}",
                    "question": f"holdout {n} {lang}",
                    "language": lang,
                    "answerable": n != 2,
                }
            )
    return questions


# --- WithinRepeatCache -------------------------------------------------------


def test_within_repeat_cache_reuses_byte_identical_prompt():
    import asyncio

    calls = []

    class _Inner:
        async def generate_structured(self, sp, up, schema, settings):
            calls.append((sp, up))
            return {"answer": "x", "citations": [], "refused": False}

    cache = gge.WithinRepeatCache(_Inner())
    schema = {"a": 1}
    asyncio.run(cache.generate_structured("S", "U", schema, _settings()))
    asyncio.run(cache.generate_structured("S", "U", schema, _settings()))
    asyncio.run(cache.generate_structured("S", "OTHER", schema, _settings()))

    assert cache.logical_calls == 3
    assert cache.forwarded_calls == 2
    assert len(calls) == 2


# --- capture_snapshots + ReplayRetriever ------------------------------------


def test_capture_snapshots_bands_and_replay_is_stable():
    questions = _holdout_questions()
    scores = {q["question"]: 0.57 for q in questions}
    scores[questions[0]["question"]] = 0.20  # hard_refuse
    scores[questions[1]["question"]] = 0.95  # confident
    snaps, replay = gge.capture_snapshots(_ScriptedRetriever(scores), questions)

    bands = {s.question_id: s.gate_band for s in snaps}
    assert bands[questions[0]["id"]] == "hard_refuse"
    assert bands[questions[1]["id"]] == "confident"
    assert bands[questions[2]["id"]] == "grounded_review"
    assert replay.retrieve(questions[1]["question"]) == replay.retrieve(questions[1]["question"])


# --- run_matrix ------------------------------------------------------------


def test_run_matrix_confident_call_is_shared_between_policies_within_repeat():
    questions = [{"id": "c0", "pair_id": "c0", "question": "q", "language": "en", "answerable": True}]
    _snaps, replay = gge.capture_snapshots(_ScriptedRetriever({"q": 0.95}), questions)
    outcomes = gge.run_matrix(questions, replay, _settings(), _fake_llm_factory(), repeats=1)

    binary = next(o for o in outcomes if o.policy == "binary")
    grounded = next(o for o in outcomes if o.policy == "grounded_review")
    assert binary.gate_band == "confident" and grounded.gate_band == "confident"
    assert binary.forwarded_calls == 1
    assert grounded.logical_calls == 1 and grounded.forwarded_calls == 0  # reused


def test_run_matrix_grey_band_only_grounded_calls_the_llm():
    questions = [{"id": "g0", "pair_id": "g0", "question": "q", "language": "en", "answerable": True}]
    _snaps, replay = gge.capture_snapshots(_ScriptedRetriever({"q": 0.57}), questions)
    outcomes = gge.run_matrix(questions, replay, _settings(), _fake_llm_factory(), repeats=1)

    binary = next(o for o in outcomes if o.policy == "binary")
    grounded = next(o for o in outcomes if o.policy == "grounded_review")
    assert binary.gate_band == "hard_refuse" and binary.logical_calls == 0
    assert grounded.gate_band == "grounded_review" and grounded.forwarded_calls == 1
    assert grounded.physical_requests == 1


def test_run_matrix_fresh_cache_per_repeat():
    questions = [{"id": "c0", "pair_id": "c0", "question": "q", "language": "en", "answerable": True}]
    _snaps, replay = gge.capture_snapshots(_ScriptedRetriever({"q": 0.95}), questions)
    outcomes = gge.run_matrix(questions, replay, _settings(), _fake_llm_factory(), repeats=2)

    forwarded_by_repeat = {}
    for o in outcomes:
        forwarded_by_repeat.setdefault(o.repeat, 0)
        forwarded_by_repeat[o.repeat] += o.forwarded_calls
    assert forwarded_by_repeat == {1: 1, 2: 1}  # each repeat forwards once, no cross-repeat reuse


# --- gates ----------------------------------------------------------------


def _outcome(**kw) -> gge.QuestionOutcome:
    base = dict(
        repeat=1,
        policy="grounded_review",
        question_id="x",
        language="en",
        answerable=True,
        refused=False,
        status="ok",
        gate_band="confident",
        decision_reason="accepted_confident",
        confidence=0.9,
        citation_count=1,
        latency_ms=1.0,
        logical_calls=1,
        forwarded_calls=1,
        physical_requests=1,
        rate_limited=0,
        repaired=0,
        schema_fallbacks=0,
        provider_fallbacks=0,
        error_type=None,
    )
    base.update(kw)
    return gge.QuestionOutcome(**base)


def test_evaluate_gates_flags_errors_and_fallbacks_and_canary():
    holdout = [_outcome(status="error", error_type="RuntimeError")]
    canary = [
        _outcome(policy="grounded_review", question_id="r001", answerable=True, refused=False, repeat=r)
        for r in (1, 2, 3)
    ]
    gates = {g.name: g for g in gge.evaluate_gates(holdout, canary)}
    assert gates["no_errors"].passed is False
    assert gates["canary_answers[r001]"].passed is True
    assert gates["canary_refuses[r018]"].passed is False  # no rows -> fails
    assert gates["citation_faithfulness_human_review"].passed is False


# --- write_run_dir -------------------------------------------------------


def test_write_run_dir_is_write_once_and_checksummed(tmp_path: Path):
    manifest = {
        "run_id": "gate_generation_eval_TEST",
        "index_profile": gge.PINNED_INDEX_PROFILE,
        "expansion_mode": gge.PINNED_EXPANSION_MODE,
        "review_floor": gge.PINNED_REVIEW_FLOOR,
        "threshold": gge.PINNED_THRESHOLD,
        "llm_provider": "groq",
        "build_commit": "abc",
        "full_repeats": 1,
        "canary_repeats": 1,
    }
    args = dict(
        run_id="gate_generation_eval_TEST",
        manifest=manifest,
        snapshots=[
            gge.RetrievalSnapshot("h0en", "q", "en", True, ["chunk-1"], 0.57, "grounded_review")
        ],
        holdout=[_outcome(question_id="h0en")],
        canary=[],
        gates=[gge.GateResult("no_errors", True, "ok")],
    )
    run_dir = gge.write_run_dir(tmp_path, **args)
    for name in (
        "run_manifest.json",
        "retrieval.jsonl",
        "outcomes.jsonl",
        "comparison.md",
        "blind_checklist.csv",
        "checksums.txt",
    ):
        assert (run_dir / name).exists()
    with pytest.raises(FileExistsError):
        gge.write_run_dir(tmp_path, **args)


# --- run() end to end with fakes --------------------------------------------


def test_run_end_to_end_with_injected_fakes(tmp_path: Path):
    questions = _holdout_questions()
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(
        json.dumps({"version": "1.0.0", "sha256": "deadbeef", "status": "frozen", "questions": questions}),
        encoding="utf-8",
    )
    regression = tmp_path / "regression.json"
    reg_queries = [
        {"id": qid, "query": f"canary {qid}", "language": "en" if i % 2 else "es"}
        for i, qid in enumerate(("r001", "r002", "r018", "r019", "r020"))
    ]
    regression.write_text(json.dumps({"version": "t", "sha256": "x", "queries": reg_queries}), encoding="utf-8")

    scores = {q["question"]: 0.57 for q in questions}
    scores.update({q["query"]: 0.57 for q in reg_queries})

    run_dir = gge.run(
        holdout_path=holdout_path,
        regression_path=regression,
        out_root=tmp_path / "out",
        retriever=_ScriptedRetriever(scores),
        llm_factory=_fake_llm_factory(),
        settings=_settings(),
        build_commit="abc123",
        full_repeats=1,
        canary_repeats=1,
        now=datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert run_dir.name == "gate_generation_eval_20260831T120000Z"
    outcomes = [json.loads(line) for line in (run_dir / "outcomes.jsonl").read_text().splitlines()]
    # 6 holdout * 2 policies * 1 repeat + 5 canary * 2 * 1
    assert len(outcomes) == 6 * 2 + 5 * 2
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["index_profile"] == "contextual-v1"
    assert manifest["review_floor"] == 0.55
