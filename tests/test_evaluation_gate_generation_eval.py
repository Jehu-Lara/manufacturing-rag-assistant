from __future__ import annotations

import csv
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
_VERBATIM = "net positive suction head required (NPSHR) is a property of the pump"


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
            "source_type": "public",
            "chunk_text": _GREY_TEXT,
        },
    )


class _ScriptedRetriever:
    def __init__(self, score_by_text: dict[str, float], chunk_by_text: dict[str, str] | None = None) -> None:
        self._score_by_text = score_by_text
        self._chunk_by_text = chunk_by_text or {}

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        return [_result(self._chunk_by_text.get(query_text, "chunk-1"), self._score_by_text[query_text])]


def _fake_llm_factory(emit_physical: bool = True):
    def factory(hook):
        class _Fake:
            async def generate_structured(self, system_prompt, user_prompt, schema, settings):
                if emit_physical:
                    hook(
                        LlmTraceEvent(
                            event="physical_request",
                            provider="groq",
                            phase="initial",
                            total_tokens=7,
                            latency_ms=12.0,
                        )
                    )
                if "evidence" in schema["properties"]:
                    return {
                        "answer": "A grounded answer.",
                        "evidence": [{"chunk_id": "chunk-1", "supporting_quote": _VERBATIM}],
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
                    "expected_answer": "gold" if n != 2 else "",
                }
            )
    return questions


# --- WithinRepeatCache -----------------------------------------------------


def test_within_repeat_cache_reuses_byte_identical_prompt():
    import asyncio

    calls = []

    class _Inner:
        async def generate_structured(self, sp, up, schema, settings):
            calls.append((sp, up))
            return {"answer": "x", "citations": [], "refused": False}

    cache = gge.WithinRepeatCache(_Inner())
    asyncio.run(cache.generate_structured("S", "U", {"a": 1}, _settings()))
    asyncio.run(cache.generate_structured("S", "U", {"a": 1}, _settings()))
    asyncio.run(cache.generate_structured("S", "OTHER", {"a": 1}, _settings()))

    assert cache.logical_calls == 3
    assert cache.forwarded_calls == 2
    assert len(calls) == 2


# --- TraceCollector counts failed attempts -------------------------------


def test_trace_collector_counts_all_physical_attempts_and_latency():
    tc = gge.TraceCollector()
    tc(LlmTraceEvent(event="physical_attempt", provider="groq", phase="initial"))
    tc(LlmTraceEvent(event="physical_failed", provider="groq", phase="initial", latency_ms=5.0))
    tc(LlmTraceEvent(event="rate_limited", provider="groq", phase="initial"))
    tc(LlmTraceEvent(event="physical_attempt", provider="groq", phase="initial"))
    tc(LlmTraceEvent(event="physical_request", provider="groq", phase="initial", latency_ms=9.0, total_tokens=3))

    assert tc.physical_attempts == 2
    assert tc.physical_failed == 1
    assert tc.physical_success == 1
    assert tc.rate_limited == 1
    assert tc.total_tokens == 3
    assert sorted(tc.llm_latencies_ms) == [5.0, 9.0]


# --- capture_snapshots ---------------------------------------------------


def test_capture_snapshots_bands_latencies_and_chunk_text():
    questions = _holdout_questions()
    scores = {q["question"]: 0.57 for q in questions}
    scores[questions[0]["question"]] = 0.20
    scores[questions[1]["question"]] = 0.95
    snaps, replay, latencies, chunk_text = gge.capture_snapshots(_ScriptedRetriever(scores), questions)

    bands = {s.question_id: s.gate_band for s in snaps}
    assert bands[questions[0]["id"]] == "hard_refuse"
    assert bands[questions[1]["id"]] == "confident"
    assert bands[questions[2]["id"]] == "grounded_review"
    assert len(latencies) == len(questions)
    assert chunk_text["chunk-1"].startswith("Net positive suction head")


# --- run_matrix --------------------------------------------------------


def test_run_matrix_confident_call_shared_between_policies_within_repeat():
    questions = [{"id": "c0", "question": "q", "language": "en", "answerable": True}]
    _s, replay, _l, _c = gge.capture_snapshots(_ScriptedRetriever({"q": 0.95}), questions)
    outcomes = gge.run_matrix(questions, replay, _settings(), _fake_llm_factory(), repeats=1)

    binary = next(o for o in outcomes if o.policy == "binary")
    grounded = next(o for o in outcomes if o.policy == "grounded_review")
    assert binary.gate_band == "confident" and grounded.gate_band == "confident"
    assert binary.forwarded_calls == 1
    assert grounded.logical_calls == 1 and grounded.forwarded_calls == 0


def test_run_matrix_grey_band_only_grounded_calls_llm_and_counts_attempts():
    questions = [{"id": "g0", "question": "q", "language": "en", "answerable": True}]
    _s, replay, _l, _c = gge.capture_snapshots(_ScriptedRetriever({"q": 0.57}), questions)
    outcomes = gge.run_matrix(questions, replay, _settings(), _fake_llm_factory(), repeats=1)

    binary = next(o for o in outcomes if o.policy == "binary")
    grounded = next(o for o in outcomes if o.policy == "grounded_review")
    assert binary.gate_band == "hard_refuse" and binary.logical_calls == 0
    assert grounded.gate_band == "grounded_review" and grounded.forwarded_calls == 1
    assert grounded.physical_success == 1
    assert grounded.llm_latencies_ms == [12.0]


# --- gates -----------------------------------------------------------


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
        cited_chunk_ids=["chunk-1"],
        expected_chunk_ids=["chunk-1"],
        answer_text="ans",
        question_wall_ms=1.0,
        llm_latencies_ms=[10.0],
        logical_calls=1,
        forwarded_calls=1,
        physical_attempts=1,
        physical_success=1,
        physical_failed=0,
        rate_limited=0,
        repaired=0,
        schema_fallbacks=0,
        provider_fallbacks=0,
        total_tokens=5,
        error_type=None,
    )
    base.update(kw)
    return gge.QuestionOutcome(**base)


def test_gates_flag_errors_rate_limit_and_unsafe_unanswerable():
    holdout = [
        _outcome(status="error", error_type="RuntimeError"),
        _outcome(rate_limited=1),
        _outcome(answerable=False, refused=False, question_id="u1"),  # unsafe
    ]
    gates = {g.name: g for g in gge.evaluate_gates(holdout, [])}
    assert gates["no_errors"].passed is False
    assert gates["no_rate_limiting"].passed is False
    assert gates["no_unsafe_unanswerable_answer"].passed is False
    assert gates["citation_faithfulness_conditional"].passed is False


def test_canary_gate_requires_expected_chunk_citation():
    canary_good = [
        _outcome(policy="grounded_review", question_id="r001", cited_chunk_ids=["chunk-1"],
                 expected_chunk_ids=["chunk-1"], repeat=r)
        for r in (1, 2, 3)
    ]
    canary_bad = [
        _outcome(policy="grounded_review", question_id="r002", cited_chunk_ids=["other"],
                 expected_chunk_ids=["chunk-1"], repeat=r)
        for r in (1, 2, 3)
    ]
    gates = {g.name: g for g in gge.evaluate_gates([], canary_good + canary_bad)}
    assert gates["canary_answers_and_cites[r001]"].passed is True
    assert gates["canary_answers_and_cites[r002]"].passed is False


def test_gates_resolve_with_imported_human_verdicts():
    v = gge.HumanVerdicts(
        graded_rows=10,
        citation_pass_rate=0.95,
        faithfulness_pass_rate=0.92,
        unsafe_unanswerable_rows=0,
        unsafe_all_safe=True,
    )
    gates = {g.name: g for g in gge.evaluate_gates([_outcome()], [], v)}
    assert gates["citation_faithfulness_conditional"].passed is True
    assert gates["no_unsafe_unanswerable_answer"].passed is True


# --- blind checklist ------------------------------------------------


def test_checklist_is_opaque_per_attempt_and_includes_unsafe_unanswerables():
    run_id = "gate_generation_eval_X"
    outcomes = [
        _outcome(repeat=1, policy="binary", question_id="h0en", answer_text="bin ans"),
        _outcome(repeat=1, policy="grounded_review", question_id="h0en", answer_text="grd ans"),
        _outcome(repeat=2, policy="binary", question_id="h0en", answer_text="bin ans r2"),
        _outcome(policy="grounded_review", answerable=False, refused=False, question_id="u9",
                 answer_text="should not have answered"),
    ]
    rows = gge._checklist_rows(run_id, outcomes, {"chunk-1": _GREY_TEXT}, {"h0en": "gold"})
    assert len(rows) == 4  # 3 answered + 1 unsafe unanswerable
    assert {r["arm"] for r in rows} == {"arm-A", "arm-B"}
    assert all("binary" not in r["arm"] and "grounded" not in r["arm"] for r in rows)
    assert any(r["answerable"] == "False" and r["refused"] == "False" for r in rows)
    assert any(r["answer"] == "grd ans" for r in rows)
    assert any(r["expected_answer"] == "gold" for r in rows)


def _blank(**over) -> dict[str, str]:
    row = {k: "" for k in gge._CHECKLIST_HEADER}
    row.update(over)
    return row


def _checklist_pair() -> list[dict[str, str]]:
    return [
        _blank(row_id="a", arm="arm-A", repeat="1", question_id="h0", language="en",
               answerable="True", refused="False", answer="ans", cited_chunk_ids="c1",
               cited_chunk_texts="t", expected_answer="gold", expected_chunk_ids="c1"),
        _blank(row_id="b", arm="arm-B", repeat="1", question_id="u1", language="es",
               answerable="False", refused="False", answer="oops"),
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gge._CHECKLIST_HEADER)
        w.writeheader()
        w.writerows(rows)


def test_import_human_verdicts_roundtrip(tmp_path: Path):
    rows = _checklist_pair()
    baseline = gge.checklist_baseline(rows)
    rows[0].update(citation_accuracy_pass="y", faithfulness_pass="y")
    rows[1].update(safe_pass="n")
    path = tmp_path / "blind_checklist.csv"
    _write_csv(path, rows)

    v = gge.import_human_verdicts(path, baseline)
    assert v.citation_pass_rate == 1.0
    assert v.unsafe_unanswerable_rows == 1
    assert v.unsafe_all_safe is False


def test_import_human_verdicts_rejects_ungraded(tmp_path: Path):
    rows = _checklist_pair()
    baseline = gge.checklist_baseline(rows)
    path = tmp_path / "blind_checklist.csv"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="not fully graded"):
        gge.import_human_verdicts(path, baseline)


def test_import_human_verdicts_rejects_deleted_row(tmp_path: Path):
    rows = _checklist_pair()
    baseline = gge.checklist_baseline(rows)
    rows[0].update(citation_accuracy_pass="y", faithfulness_pass="y")
    path = tmp_path / "blind_checklist.csv"
    _write_csv(path, rows[:1])  # unsafe row 'b' deleted
    with pytest.raises(ValueError, match="does not match the sealed baseline"):
        gge.import_human_verdicts(path, baseline)


def test_import_human_verdicts_rejects_added_or_duplicate_row(tmp_path: Path):
    rows = _checklist_pair()
    baseline = gge.checklist_baseline(rows)
    for r in rows:
        r.update(citation_accuracy_pass="y", faithfulness_pass="y", safe_pass="y")
    path = tmp_path / "blind_checklist.csv"
    _write_csv(path, [*rows, dict(rows[0])])  # duplicate row_id 'a'
    with pytest.raises(ValueError, match="duplicate row_id"):
        gge.import_human_verdicts(path, baseline)


def test_import_human_verdicts_rejects_altered_immutable_field(tmp_path: Path):
    rows = _checklist_pair()
    baseline = gge.checklist_baseline(rows)
    rows[0].update(citation_accuracy_pass="y", faithfulness_pass="y", answer="doctored answer")
    rows[1].update(safe_pass="y")
    path = tmp_path / "blind_checklist.csv"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="altered immutable column"):
        gge.import_human_verdicts(path, baseline)


# --- write_run_dir atomic -------------------------------------------


def _write_args(tmp_path: Path):
    return dict(
        run_id="gate_generation_eval_TEST",
        manifest={
            "run_id": "gate_generation_eval_TEST",
            "index_profile": gge.PINNED_INDEX_PROFILE,
            "expansion_mode": gge.PINNED_EXPANSION_MODE,
            "review_floor": gge.PINNED_REVIEW_FLOOR,
            "threshold": gge.PINNED_THRESHOLD,
            "llm_provider": "groq",
            "build_commit": "abc",
            "full_repeats": 1,
            "canary_repeats": 1,
        },
        snapshots=[gge.RetrievalSnapshot("h0en", "q", "en", True, ["chunk-1"], 0.57, "grounded_review")],
        holdout=[_outcome(question_id="h0en")],
        canary=[],
        gates=[gge.GateResult("no_errors", True, "ok")],
        checklist_rows=[],
        arm_map={"arm-A": "binary", "arm-B": "grounded_review"},
    )


def test_write_run_dir_is_atomic_write_once_and_checksummed(tmp_path: Path):
    run_dir = gge.write_run_dir(tmp_path, **_write_args(tmp_path))
    for name in (
        "run_manifest.json",
        "retrieval.jsonl",
        "outcomes.jsonl",
        "comparison.md",
        "blind_checklist.csv",
        "blind_checklist.baseline.json",
        "arm_map.sealed.json",
        "checksums.txt",
    ):
        assert (run_dir / name).exists()
    assert not (tmp_path / "gate_generation_eval_TEST.partial").exists()
    with pytest.raises(FileExistsError):
        gge.write_run_dir(tmp_path, **_write_args(tmp_path))


def test_cli_import_verdicts_does_not_require_provider(tmp_path: Path, monkeypatch):
    called = {}
    monkeypatch.setattr(gge, "import_verdicts_into_run", lambda run_dir: called.setdefault("dir", run_dir))
    gge.main(["--import-verdicts", str(tmp_path / "somerun")])
    assert called["dir"] == tmp_path / "somerun"


def test_cli_paid_run_requires_provider():
    with pytest.raises(SystemExit):
        gge.main(["--full-repeats", "1"])


# --- run() end to end -------------------------------------------------


def test_assert_profile_coverage_requires_min_grey_per_cell():
    snaps: list[gge.RetrievalSnapshot] = []
    for lang in ("en", "es"):
        for answerable in (True, False):
            for n in range(gge.gate_holdout_profile.MIN_GREY_PER_CELL):
                snaps.append(
                    gge.RetrievalSnapshot(
                        f"{lang}{answerable}{n}", "q", lang, answerable, ["c"], 0.57, "grounded_review"
                    )
                )
    gge._assert_profile_coverage(snaps)  # exactly the minimum -> ok

    thin = snaps[:-1]  # one es/unanswerable question short
    with pytest.raises(RuntimeError, match="grounded-review band"):
        gge._assert_profile_coverage(thin)


def test_verify_prereqs_rejects_provider_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(gge.eval_set_integrity, "verify", lambda *a, **k: None)
    monkeypatch.setattr(gge.regression_set_integrity, "verify", lambda *a, **k: None)
    monkeypatch.setattr(gge.gate_holdout_integrity, "verify", lambda *a, **k: None)
    monkeypatch.setattr(gge, "assert_live_index_profile", lambda p: None)

    class _M:
        index_profile = "contextual-v1"
        build_commit = "abc"

    monkeypatch.setattr(gge.index_manifest, "read", lambda: _M())
    monkeypatch.setattr(gge, "load_settings", lambda: _settings())  # llm_provider="groq"
    with pytest.raises(RuntimeError, match="does not match LLM_PROVIDER"):
        gge._verify_prereqs(tmp_path / "h.json", tmp_path / "r.json", "openai")


def test_run_end_to_end_with_injected_fakes(tmp_path: Path):
    questions = _holdout_questions()
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(
        json.dumps({"version": "1.0.0", "sha256": "deadbeef", "status": "frozen", "questions": questions}),
        encoding="utf-8",
    )
    reg_queries = [
        {"id": qid, "query": f"canary {qid}", "language": "en" if i % 2 else "es",
         "expected_chunk_id": "chunk-1" if qid in ("r001", "r002") else None}
        for i, qid in enumerate(("r001", "r002", "r018", "r019", "r020"))
    ]
    regression = tmp_path / "regression.json"
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
    assert len(outcomes) == 6 * 2 + 5 * 2
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["index_profile"] == "contextual-v1"
    assert manifest["verdicts_imported"] is False
    arm_map = json.loads((run_dir / "arm_map.sealed.json").read_text())
    assert set(arm_map.values()) == {"binary", "grounded_review"}

    # import_verdicts flow
    checklist = run_dir / "blind_checklist.csv"
    with checklist.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["citation_accuracy_pass"] = "y"
        r["faithfulness_pass"] = "y"
        r["safe_pass"] = "y"
    with checklist.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gge._CHECKLIST_HEADER)
        w.writeheader()
        w.writerows(rows)
    gge.import_verdicts_into_run(run_dir)
    assert json.loads((run_dir / "run_manifest.json").read_text())["verdicts_imported"] is True
    assert "citation=" in (run_dir / "comparison.md").read_text()
