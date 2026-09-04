from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.adapters.secondary.llm.groq_openai_client import GroqOpenAiLlmClient, TraceHook
from src.core.config import (
    Settings,
    load_settings,
)
from src.core.paths import EVAL_REPORTS_DIR
from src.domain.ports import LLMClientPort, RetrieverPort
from src.features.evaluation import (
    eval_set_integrity,
    gate_holdout_integrity,
    gate_holdout_profile,
    regression_set_integrity,
)
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.evaluation.gate_eval.artifacts import (
    _CHECKLIST_HEADER,
    _EDITABLE_COLUMNS,
    _IMMUTABLE_COLUMNS,
    _VERDICT_COLUMNS,
    _arm_labels,
    _checklist_rows,
    _finalize_dir,
    _percentiles,
    _sha256_file,
    _write_checklist,
    _write_jsonl,
    checklist_baseline,
    render_comparison,
    write_run_dir,
)
from src.features.evaluation.gate_eval.models import (
    _POLICIES,
    CANARY_MUST_ANSWER,
    CANARY_MUST_REFUSE,
    CANARY_REPEATS,
    FULL_REPEATS,
    PINNED_EXPANSION_MODE,
    PINNED_INDEX_PROFILE,
    PINNED_REVIEW_FLOOR,
    PINNED_THRESHOLD,
    GateResult,
    QuestionOutcome,
    ReplayRetriever,
    RetrievalSnapshot,
    TraceCollector,
    WithinRepeatCache,
    _band,
    _schema_key,
)

# Implementation split out on 2026-09-04 (audit bucket 4). This module stays
# the compatibility facade: the CLI surface, every public name, and the run
# artifact contract are unchanged.
from src.features.evaluation.gate_eval.runner import (
    _lang,
    _use_case,
    capture_snapshots,
    run_matrix,
)
from src.features.evaluation.gate_eval.verdicts import (
    HumanVerdicts,
    _parse_pass,
    _rate,
    _verify_against_baseline,
    evaluate_gates,
    import_human_verdicts,
)
from src.features.retrieval import index_manifest

REPORT_ROOT = EVAL_REPORTS_DIR

# Names re-exported for the CLI, the test suite and any sealed-run tooling that
# imports them from here. Listed explicitly so the linter does not prune a
# re-export this module itself never calls.
__all__ = [
    "HumanVerdicts",
    "_lang",
    "_parse_pass",
    "_rate",
    "_use_case",
    "_verify_against_baseline",
    "import_human_verdicts",
    # Re-exported implementation names. Underscore-prefixed ones are listed
    # deliberately: the test suite and the sealed-run tooling reach for them
    # through this facade, and without __all__ the linter prunes a re-export
    # this module never calls itself.
    "_CHECKLIST_HEADER",
    "_EDITABLE_COLUMNS",
    "_IMMUTABLE_COLUMNS",
    "_VERDICT_COLUMNS",
    "_arm_labels",
    "_band",
    "_checklist_rows",
    "_finalize_dir",
    "_percentiles",
    "_schema_key",
    "_sha256_file",
    "_write_checklist",
    "_write_jsonl",
    "checklist_baseline",
    "render_comparison",
    "CANARY_MUST_ANSWER",
    "CANARY_MUST_REFUSE",
    "CANARY_REPEATS",
    "FULL_REPEATS",
    "PINNED_EXPANSION_MODE",
    "PINNED_INDEX_PROFILE",
    "PINNED_REVIEW_FLOOR",
    "PINNED_THRESHOLD",
    "REPORT_ROOT",
    "GateResult",
    "QuestionOutcome",
    "ReplayRetriever",
    "RetrievalSnapshot",
    "TraceCollector",
    "WithinRepeatCache",
    "capture_snapshots",
    "evaluate_gates",
    "import_verdicts_into_run",
    "main",
    "run",
    "run_matrix",
    "write_run_dir",
]


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
            return GroqOpenAiLlmClient.from_settings(
                settings, allow_provider_fallback=False, trace_hook=hook
            )

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

    gates = evaluate_gates(
        holdout_outcomes,
        canary_outcomes,
        expected_holdout_ids=frozenset(q["id"] for q in holdout_questions),
        expected_canary_ids=frozenset(q["id"] for q in canary_questions),
    )
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
    """Re-grade the gates that need the human blind review. Single-use and
    append-only: five sealed files (run_manifest.json, outcomes.jsonl,
    retrieval.jsonl, blind_checklist.baseline.json, arm_map.sealed.json)
    are hash-verified, never rewritten (comparison.md is likewise never
    rewritten); verdict state goes to import_manifest.json and the
    re-rendered comparison.import.md, with checksums.import.txt written
    last as the completion marker."""
    if (run_dir / "checksums.import.txt").exists():
        raise ValueError(
            f"{run_dir} already has checksums.import.txt - verdict imports are single-use; "
            "re-running would overwrite the import artifacts and re-seal them"
        )
    checksums_file = run_dir / "checksums.txt"
    if not checksums_file.exists():
        raise ValueError(
            f"{run_dir} has no checksums.txt - refusing to import verdicts "
            "without the sealed hashes to verify against"
        )
    sealed_hashes: dict[str, str] = {}
    for lineno, line in enumerate(checksums_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"{checksums_file} line {lineno} is malformed - expected '<sha256>  <filename>'"
            )
        sealed_hashes[parts[1]] = parts[0]
    immutable_files = (
        "run_manifest.json",
        "outcomes.jsonl",
        "retrieval.jsonl",
        "blind_checklist.baseline.json",
        "arm_map.sealed.json",
    )
    for immutable_name in immutable_files:
        target_path = run_dir / immutable_name
        if immutable_name not in sealed_hashes:
            raise ValueError(
                f"{immutable_name} has no sealed hash in {checksums_file} - refusing to import verdicts"
            )
        if not target_path.exists():
            raise ValueError(
                f"{immutable_name} is missing from {run_dir} - refusing to import verdicts"
            )
        actual_hash = _sha256_file(target_path)
        if actual_hash != sealed_hashes[immutable_name]:
            raise ValueError(
                f"{immutable_name} hash {actual_hash} mismatch against sealed checksums.txt "
                f"{sealed_hashes[immutable_name]} - the run directory was tampered with"
            )

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
    canary_ids = set((*CANARY_MUST_ANSWER, *CANARY_MUST_REFUSE))
    gates = evaluate_gates(
        holdout,
        canary,
        verdicts,
        expected_holdout_ids=frozenset(s.question_id for s in snapshots if s.question_id not in canary_ids),
        expected_canary_ids=frozenset(canary_ids),
    )
    render_manifest = {**manifest, "verdicts_imported": True}
    (run_dir / "import_manifest.json").write_text(
        json.dumps({"verdicts_imported": True, **asdict(verdicts)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "comparison.import.md").write_text(
        render_comparison(render_manifest, snapshots, holdout, canary, gates), encoding="utf-8"
    )
    imported_checksums = [
        f"{_sha256_file(p)}  {p.name}"
        for p in sorted(run_dir.iterdir())
        if p.name not in ("checksums.txt", "checksums.import.txt") and p.is_file()
    ]
    (run_dir / "checksums.import.txt").write_text("\n".join(imported_checksums) + "\n", encoding="utf-8")
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
