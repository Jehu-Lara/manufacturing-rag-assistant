from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.features.evaluation.gate_eval.artifacts import (
    _CHECKLIST_HEADER,
    _IMMUTABLE_COLUMNS,
    _VERDICT_COLUMNS,
)
from src.features.evaluation.gate_eval.models import (
    _POLICIES,
    CANARY_MUST_ANSWER,
    CANARY_MUST_REFUSE,
    CANARY_REPEATS,
    FULL_REPEATS,
    GateResult,
    QuestionOutcome,
)


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
    expected_holdout_ids: frozenset[str] | None = None,
    expected_canary_ids: frozenset[str] | None = None,
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
                f"(gate requires {CANARY_REPEATS}/{CANARY_REPEATS}; entailment still needs blind grading)",
            )
        )
    for qid in CANARY_MUST_REFUSE:
        hits = [r for r in grounded_canary if r.question_id == qid]
        refused = sum(r.refused for r in hits)
        gates.append(
            GateResult(
                f"canary_refuses[{qid}]",
                len(hits) == CANARY_REPEATS and refused == CANARY_REPEATS,
                f"{refused}/{len(hits)} refused (gate requires {CANARY_REPEATS}/{CANARY_REPEATS})",
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

    full_repeats = FULL_REPEATS
    canary_repeats = CANARY_REPEATS
    repeat_problems: list[str] = []
    for label, rows, repeats, sealed in (
        ("holdout", holdout, FULL_REPEATS, expected_holdout_ids),
        ("canary", canary, CANARY_REPEATS, expected_canary_ids),
    ):
        want = list(range(1, repeats + 1))
        cells: dict[str, list[QuestionOutcome]] = {}
        for row in rows:
            cells.setdefault(row.question_id, []).append(row)
        if sealed is not None:
            missing = sorted(set(sealed) - set(cells))
            extra = sorted(set(cells) - set(sealed))
            if missing or extra:
                repeat_problems.append(f"{label}: question set mismatch - missing {missing}, extra {extra}")
        if not cells:
            repeat_problems.append(f"{label}: no outcomes")
            continue
        for qid in sorted(cells):
            for policy in _POLICIES:
                got = sorted(r.repeat for r in cells[qid] if r.policy == policy)
                if got != want:
                    repeat_problems.append(f"{label}/{qid}/{policy}: repeats {got} != {want}")
    repeats_ok = not repeat_problems
    repeats_detail = (
        "; ".join(repeat_problems)
        if repeat_problems
        else f"holdout x{full_repeats} (needs {FULL_REPEATS}), canary x{canary_repeats} (needs {CANARY_REPEATS})"
    )
    gates.append(
        GateResult(
            "repeats_conform",
            repeats_ok,
            repeats_detail,
        )
    )
    return gates
