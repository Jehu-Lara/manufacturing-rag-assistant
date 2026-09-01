from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from src.domain.models import ExpansionMode, IndexProfile
from src.features.evaluation import artifacts, eval_set_integrity, metrics
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K, HybridRetriever

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"

SWEEP_STEP = 0.02

_LANGUAGES: list[tuple[str, str]] = [("en", "English"), ("es", "Spanish")]

DIAGNOSTIC_NOTE = (
    "Per-language tables are diagnostic; the shipped threshold remains the "
    "pooled selection (0.5999 override, SPEC.md Phase 3)."
)


def _top1_semantic_score(retriever: HybridRetriever, question_text: str) -> float:
    results = retriever.retrieve(question_text, k=SEMANTIC_EXTRACTION_K)
    return metrics.top1_semantic_score(results)


def sweep_thresholds(low: float, high: float, step: float = SWEEP_STEP) -> list[float]:
    if high < low:
        low, high = high, low
    steps = max(int(round((high - low) / step)), 0)
    return [round(low + i * step, 10) for i in range(steps + 1)]


def _refusal_counts(
    threshold: float, unanswerable_scores: list[float], answerable_scores: list[float]
) -> tuple[int, int]:
    answerable_wrongly_refused = sum(1 for score in answerable_scores if score < threshold)
    unanswerable_correctly_refused = sum(1 for score in unanswerable_scores if score < threshold)
    return answerable_wrongly_refused, unanswerable_correctly_refused


def select_threshold(
    unanswerable_scores: list[float], answerable_scores: list[float], step: float = SWEEP_STEP
) -> dict[str, Any]:
    """- no-overlap ranges: midpoint between max(unanswerable) and min(answerable).
    - overlapping ranges: sweep-table threshold maximizing
      (unanswerable_correctly_refused - answerable_wrongly_refused), ties broken
      toward the LOWEST candidate threshold (a false refusal is treated as worse
      than an answer-attempt a later faithfulness review would catch).
    """
    if not unanswerable_scores or not answerable_scores:
        raise ValueError("both score lists must be non-empty")

    unanswerable_min = min(unanswerable_scores)
    unanswerable_max = max(unanswerable_scores)
    answerable_min = min(answerable_scores)
    answerable_max = max(answerable_scores)

    sweep_low = unanswerable_min - step
    sweep_high = answerable_max + step
    candidates = sweep_thresholds(sweep_low, sweep_high, step)

    sweep = []
    for candidate in candidates:
        answerable_wrongly_refused, unanswerable_correctly_refused = _refusal_counts(
            candidate, unanswerable_scores, answerable_scores
        )
        sweep.append(
            {
                "threshold": candidate,
                "answerable_wrongly_refused": answerable_wrongly_refused,
                "unanswerable_correctly_refused": unanswerable_correctly_refused,
                "objective": unanswerable_correctly_refused - answerable_wrongly_refused,
            }
        )

    if unanswerable_max < answerable_min:
        branch = "no_overlap"
        threshold = (unanswerable_max + answerable_min) / 2
    else:
        branch = "overlap"
        best_objective = max(row["objective"] for row in sweep)
        best_candidates = [row["threshold"] for row in sweep if row["objective"] == best_objective]
        threshold = min(best_candidates)

    return {
        "branch": branch,
        "threshold": threshold,
        "sweep": sweep,
        "unanswerable_min": unanswerable_min,
        "unanswerable_max": unanswerable_max,
        "answerable_min": answerable_min,
        "answerable_max": answerable_max,
    }


def _stats_line(scores: list[float]) -> str:
    return (
        f"min={min(scores):.4f}, max={max(scores):.4f}, "
        f"mean={statistics.mean(scores):.4f}, median={statistics.median(scores):.4f}"
    )


def _filter_by_language(scores: list[float], languages: list[str], target: str) -> list[float]:
    return [score for score, language in zip(scores, languages, strict=True) if language == target]


def _language_sections(
    unanswerable_scores: list[float],
    answerable_scores: list[float],
    unanswerable_languages: list[str],
    answerable_languages: list[str],
) -> list[str]:
    lines: list[str] = [
        "## Per-language diagnostics",
        "",
        DIAGNOSTIC_NOTE,
        "",
    ]
    for code, label in _LANGUAGES:
        lang_unanswerable = _filter_by_language(unanswerable_scores, unanswerable_languages, code)
        lang_answerable = _filter_by_language(answerable_scores, answerable_languages, code)

        lines += [
            f"## {label} — answerable/unanswerable top-1 semantic_score",
            "",
            f"- Unanswerable (n={len(lang_unanswerable)}): sorted={sorted(lang_unanswerable)}",
        ]
        if lang_unanswerable:
            lines.append(f"  - Stats: {_stats_line(lang_unanswerable)}")
        lines.append(f"- Answerable (n={len(lang_answerable)}): sorted={sorted(lang_answerable)}")
        if lang_answerable:
            lines.append(f"  - Stats: {_stats_line(lang_answerable)}")

        lines += ["", f"## {label} — cutoff sweep", ""]
        if not lang_unanswerable or not lang_answerable:
            lines += ["_Not enough per-language data to sweep._", ""]
            continue

        lines += [
            "| threshold | answerable wrongly refused | unanswerable correctly refused | objective (correct - wrong) |",
            "|---|---|---|---|",
        ]
        sweep_low = min(lang_unanswerable) - SWEEP_STEP
        sweep_high = max(lang_answerable) + SWEEP_STEP
        for candidate in sweep_thresholds(sweep_low, sweep_high):
            wrongly_refused, correctly_refused = _refusal_counts(
                candidate, lang_unanswerable, lang_answerable
            )
            lines.append(
                f"| {candidate:.4f} | {wrongly_refused} | {correctly_refused} | "
                f"{correctly_refused - wrongly_refused} |"
            )
        lines.append("")
    return lines


def build_report(
    unanswerable_scores: list[float],
    answerable_scores: list[float],
    unanswerable_languages: list[str],
    answerable_languages: list[str],
    selection: dict[str, Any],
    version: str,
) -> str:
    lines = [
        f"# Refusal Threshold Analysis — eval_set v{version}",
        "",
        (
            "Computed from real hybrid-retrieval runs over the hash-verified eval set. "
            "Scores are each question's top-1 *pure-semantic* cosine similarity "
            "(`RetrievalResult.semantic_score` at `semantic_rank == 1`), "
            "not the RRF `fused_score` — fused_score is disqualified as a refusal-"
            "confidence signal (rank-based, not magnitude-based)."
        ),
        "",
        "## Unanswerable subset — top-1 semantic_score (n=%d)" % len(unanswerable_scores),
        "",
        f"- Sorted: {sorted(unanswerable_scores)}",
        f"- Stats: {_stats_line(unanswerable_scores)}",
        "",
        "## Answerable subset — top-1 semantic_score (n=%d)" % len(answerable_scores),
        "",
        f"- Sorted: {sorted(answerable_scores)}",
        f"- Stats: {_stats_line(answerable_scores)}",
        "",
        "## Cutoff sweep",
        "",
        "Refusal rule under test at each candidate threshold: refuse when top-1 "
        "semantic_score < threshold.",
        "",
        "| threshold | answerable wrongly refused | unanswerable correctly refused | objective (correct - wrong) |",
        "|---|---|---|---|",
    ]
    for row in selection["sweep"]:
        lines.append(
            f"| {row['threshold']:.4f} | {row['answerable_wrongly_refused']} | "
            f"{row['unanswerable_correctly_refused']} | {row['objective']} |"
        )

    lines += ["", "## Selection procedure and result", ""]
    if selection["branch"] == "no_overlap":
        lines += [
            "- **Branch taken: no-overlap.** "
            f"max(unanswerable top-1 semantic_score) = {selection['unanswerable_max']:.4f} "
            f"< min(answerable top-1 semantic_score) = {selection['answerable_min']:.4f} "
            "— the two ranges do not overlap.",
            "- Rule applied: threshold = midpoint between those two values.",
            (
                f"- Chosen threshold = ({selection['unanswerable_max']:.4f} + "
                f"{selection['answerable_min']:.4f}) / 2 = **{selection['threshold']:.4f}**."
            ),
        ]
    else:
        lines += [
            "- **Branch taken: overlap.** "
            f"max(unanswerable top-1 semantic_score) = {selection['unanswerable_max']:.4f} "
            f">= min(answerable top-1 semantic_score) = {selection['answerable_min']:.4f} "
            "— the two ranges overlap.",
            (
                "- Rule applied: pick the sweep-table threshold that maximizes "
                "(unanswerable_correctly_refused - answerable_wrongly_refused); ties broken "
                "toward the lowest candidate threshold."
            ),
            f"- Chosen threshold = **{selection['threshold']:.4f}**.",
        ]
    lines += [
        "",
        (
            "**Analyzer-selected threshold on this eval set (diagnostic only — NOT applied; "
            f"production REFUSAL_COSINE_THRESHOLD stays 0.5999): {selection['threshold']:.4f}**"
        ),
        "",
    ]

    lines += _language_sections(
        unanswerable_scores, answerable_scores, unanswerable_languages, answerable_languages
    )

    return "\n".join(lines) + "\n"


def run(
    expansion_mode: ExpansionMode = "off",
    *,
    index_profile: IndexProfile = "raw-v1",
    write_canonical_alias: bool = False,
) -> Path:
    if write_canonical_alias:
        artifacts.ensure_canonical_alias_allowed(index_profile, expansion_mode)

    eval_set_integrity.verify()
    data = eval_set_integrity.load_eval_set()
    questions = data["questions"]

    assert_live_index_profile(index_profile)
    retriever = build_retriever(expansion_mode, expected_profile=index_profile)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    answerable_scores = [_top1_semantic_score(retriever, q["question"]) for q in answerable]
    unanswerable_scores = [_top1_semantic_score(retriever, q["question"]) for q in unanswerable]
    answerable_languages = [str(q["language"]) for q in answerable]
    unanswerable_languages = [str(q["language"]) for q in unanswerable]

    selection = select_threshold(unanswerable_scores, answerable_scores)

    version = data["version"]
    header = artifacts.resolve_provenance(index_profile, expansion_mode)
    report = header.render() + "\n\n" + build_report(
        unanswerable_scores,
        answerable_scores,
        unanswerable_languages,
        answerable_languages,
        selection,
        version,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / artifacts.artifact_filename(
        "threshold_analysis", version, index_profile, expansion_mode, "md"
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    if write_canonical_alias:
        (REPORT_DIR / f"threshold_analysis_v{version}.md").write_text(
            report, encoding="utf-8", newline="\n"
        )

    print(
        "Analyzer-selected threshold (diagnostic only, NOT applied): "
        f"{selection['threshold']:.4f}"
    )
    print(f"Selection branch: {selection['branch']}")
    print(f"Report written to: {report_path}")

    return report_path


if __name__ == "__main__":
    run()
