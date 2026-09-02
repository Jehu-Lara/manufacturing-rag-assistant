from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Literal, Sequence

from src.core.config import (
    DEFAULT_REFUSAL_COSINE_THRESHOLD,
    DEFAULT_REFUSAL_REVIEW_FLOOR,
)
from src.domain.models import GateBand, IndexProfile, RetrievalResult
from src.domain.policies import RefusalPolicy
from src.features.evaluation import artifacts, eval_set_integrity, regression_set_integrity
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.retrieval import index_manifest
from src.features.retrieval.use_cases import DEFAULT_TOP_N, SEMANTIC_EXTRACTION_K

RETRIEVE_K = SEMANTIC_EXTRACTION_K
CHANNEL_TOP_N = DEFAULT_TOP_N
SIGNAL_TOP_N = 10
THRESHOLD = DEFAULT_REFUSAL_COSINE_THRESHOLD
CURRENT_FLOOR = DEFAULT_REFUSAL_REVIEW_FLOOR
FLOORS: tuple[float, ...] = (0.50, 0.51, 0.52, 0.53, 0.54, 0.55)
SIGNAL_NAMES: tuple[str, ...] = (
    "none",
    "sem_top1_in_bm25_top_n",
    "sem_bm25_top1_agree",
    "channels_overlap_top_n",
)
CURRENT_RULE = (CURRENT_FLOOR, "none")
MAX_NEWLY_REVIEWED_POOLED = 2
MAX_NEWLY_REVIEWED_PER_LANGUAGE = 1
PINNED_CONTROLS: tuple[tuple[str, GateBand], ...] = (
    ("r001", "grounded_review"),
    ("r002", "grounded_review"),
    ("r018", "grounded_review"),
    ("r019", "hard_refuse"),
    ("r020", "hard_refuse"),
)

CALIBRATION_DIR = Path(__file__).resolve().parents[3] / "eval" / "calibration"
_BANNER = (
    "NOT APPLIED — mechanically eligible candidates only. Rule selection, the ADR-009 "
    "amendment, and the default flip are owner decisions gated on a fresh confirmatory "
    "holdout (v2)."
)

Cohort = Literal["eval_set", "regression_controls"]


@dataclass(frozen=True)
class FeatureRow:
    question_id: str
    language: str
    answerable: bool
    cohort: Cohort
    top1_semantic: float | None
    sem_margin: float | None
    sem_top1_id: str | None
    bm25_top1_id: str | None
    expected_sem_rank: int | None
    expected_bm25_rank: int | None
    sem_bm25_top1_agree: bool
    sem_top1_in_bm25_top_n: bool
    channels_overlap_top_n: bool


@dataclass(frozen=True)
class CellStat:
    floor: float
    signal: str
    cohort: Cohort
    language: str
    answerable: bool
    n: int
    hard_refuse: int
    grounded_review: int
    confident: int
    answerable_wrongly_hard_refused: int
    newly_reviewed_unanswerable: int


@dataclass(frozen=True)
class RuleAssessment:
    floor: float
    signal: str
    g1_fidelity: bool
    g2_false_refusal_reduction: bool
    g3_unanswerable_containment: bool
    g4_regression_controls: bool

    @property
    def eligible(self) -> bool:
        return all(
            (
                self.g1_fidelity,
                self.g2_false_refusal_reduction,
                self.g3_unanswerable_containment,
                self.g4_regression_controls,
            )
        )


def _ranked_ids(results: Sequence[RetrievalResult], rank_attr: str) -> list[str]:
    ranked = [result for result in results if getattr(result, rank_attr) is not None]
    ranked.sort(key=lambda result: int(getattr(result, rank_attr)))
    return [result.chunk_id for result in ranked]


def _semantic_score(results: Sequence[RetrievalResult], rank: int) -> float | None:
    for result in results:
        if result.semantic_rank == rank:
            return result.semantic_score
    return None


def _best_expected_rank(ranked_ids: Sequence[str], expected_chunk_ids: Sequence[str]) -> int | None:
    positions = [ranked_ids.index(chunk_id) + 1 for chunk_id in expected_chunk_ids if chunk_id in ranked_ids]
    return min(positions) if positions else None


def extract_features(
    results: Sequence[RetrievalResult],
    *,
    question_id: str,
    language: str,
    answerable: bool,
    cohort: Cohort,
    expected_chunk_ids: Sequence[str],
) -> FeatureRow:
    semantic_ids = _ranked_ids(results, "semantic_rank")[:CHANNEL_TOP_N]
    bm25_ids = _ranked_ids(results, "bm25_rank")[:CHANNEL_TOP_N]
    score1 = _semantic_score(results, 1)
    score2 = _semantic_score(results, 2)
    sem_top1_id = semantic_ids[0] if semantic_ids else None
    bm25_top1_id = bm25_ids[0] if bm25_ids else None
    bm25_top_n = set(bm25_ids[:SIGNAL_TOP_N])
    sem_top_n = set(semantic_ids[:SIGNAL_TOP_N])
    return FeatureRow(
        question_id=question_id,
        language=language,
        answerable=answerable,
        cohort=cohort,
        top1_semantic=score1,
        sem_margin=None if score1 is None or score2 is None else score1 - score2,
        sem_top1_id=sem_top1_id,
        bm25_top1_id=bm25_top1_id,
        expected_sem_rank=_best_expected_rank(semantic_ids, expected_chunk_ids),
        expected_bm25_rank=_best_expected_rank(bm25_ids, expected_chunk_ids),
        sem_bm25_top1_agree=sem_top1_id is not None and sem_top1_id == bm25_top1_id,
        sem_top1_in_bm25_top_n=sem_top1_id is not None and sem_top1_id in bm25_top_n,
        channels_overlap_top_n=bool(sem_top_n & bm25_top_n),
    )


def _signal_passes(row: FeatureRow, signal: str) -> bool:
    if signal == "none":
        return True
    if signal == "sem_top1_in_bm25_top_n":
        return row.sem_top1_in_bm25_top_n
    if signal == "sem_bm25_top1_agree":
        return row.sem_bm25_top1_agree
    if signal == "channels_overlap_top_n":
        return row.channels_overlap_top_n
    raise ValueError(f"unknown entry signal {signal!r}; expected one of {SIGNAL_NAMES}")


def classify_candidate(row: FeatureRow, floor: float, signal: str) -> GateBand:
    score = row.top1_semantic
    if score is None:
        return "hard_refuse"
    if score >= THRESHOLD:
        return "confident"
    if score < floor:
        return "hard_refuse"
    return "grounded_review" if _signal_passes(row, signal) else "hard_refuse"


def sweep_grid(rows: Sequence[FeatureRow]) -> list[CellStat]:
    current_floor, current_signal = CURRENT_RULE
    slices = sorted({(row.cohort, row.language, row.answerable) for row in rows})
    stats: list[CellStat] = []
    for floor, signal in product(FLOORS, SIGNAL_NAMES):
        for cohort, language, answerable in slices:
            members = [
                row
                for row in rows
                if (row.cohort, row.language, row.answerable) == (cohort, language, answerable)
            ]
            bands = [classify_candidate(row, floor, signal) for row in members]
            current = [classify_candidate(row, current_floor, current_signal) for row in members]
            stats.append(
                CellStat(
                    floor=floor,
                    signal=signal,
                    cohort=cohort,
                    language=language,
                    answerable=answerable,
                    n=len(members),
                    hard_refuse=bands.count("hard_refuse"),
                    grounded_review=bands.count("grounded_review"),
                    confident=bands.count("confident"),
                    answerable_wrongly_hard_refused=(
                        bands.count("hard_refuse") if answerable else 0
                    ),
                    newly_reviewed_unanswerable=(
                        0
                        if answerable
                        else sum(
                            1
                            for before, after in zip(current, bands, strict=True)
                            if before == "hard_refuse" and after == "grounded_review"
                        )
                    ),
                )
            )
    return stats


def _fidelity(rows: Sequence[FeatureRow]) -> bool:
    policy = RefusalPolicy(THRESHOLD, mode="grounded_review", review_floor=CURRENT_FLOOR)
    return all(
        classify_candidate(row, *CURRENT_RULE) == policy.classify_score(row.top1_semantic)
        for row in rows
    )


def _stat_cells(
    stats: Sequence[CellStat], floor: float, signal: str, cohort: Cohort
) -> list[CellStat]:
    return [
        stat
        for stat in stats
        if stat.floor == floor and stat.signal == signal and stat.cohort == cohort
    ]


def _wrongly_refused_by_language(
    stats: Sequence[CellStat], floor: float, signal: str
) -> dict[str, int]:
    return {
        stat.language: stat.answerable_wrongly_hard_refused
        for stat in _stat_cells(stats, floor, signal, "eval_set")
        if stat.answerable
    }


def assess_rule(
    rows: Sequence[FeatureRow], stats: Sequence[CellStat], floor: float, signal: str
) -> RuleAssessment:
    current_wrong = _wrongly_refused_by_language(stats, *CURRENT_RULE)
    candidate_wrong = _wrongly_refused_by_language(stats, floor, signal)
    languages = set(current_wrong) | set(candidate_wrong)
    g2 = bool(languages) and any(
        candidate_wrong.get(language, 0) < current_wrong.get(language, 0)
        for language in languages
    ) and all(
        candidate_wrong.get(language, 0) <= current_wrong.get(language, 0)
        for language in languages
    )

    promotion_cells = [
        stat
        for stat in _stat_cells(stats, floor, signal, "eval_set")
        if not stat.answerable
    ]
    g3 = sum(stat.newly_reviewed_unanswerable for stat in promotion_cells) <= (
        MAX_NEWLY_REVIEWED_POOLED
    ) and all(
        stat.newly_reviewed_unanswerable <= MAX_NEWLY_REVIEWED_PER_LANGUAGE
        for stat in promotion_cells
    )

    controls = {row.question_id: row for row in rows if row.cohort == "regression_controls"}
    pinned = dict(PINNED_CONTROLS)
    g4 = pinned.keys() <= controls.keys()
    if g4:
        for row in controls.values():
            candidate = classify_candidate(row, floor, signal)
            current = classify_candidate(row, *CURRENT_RULE)
            if row.answerable and current != "hard_refuse" and candidate == "hard_refuse":
                g4 = False
            if not row.answerable and current == "hard_refuse" and candidate != "hard_refuse":
                g4 = False
            if row.question_id in pinned and candidate != pinned[row.question_id]:
                g4 = False

    return RuleAssessment(
        floor=floor,
        signal=signal,
        g1_fidelity=_fidelity(rows),
        g2_false_refusal_reduction=g2,
        g3_unanswerable_containment=g3,
        g4_regression_controls=g4,
    )


def assess_grid(rows: Sequence[FeatureRow], stats: Sequence[CellStat]) -> list[RuleAssessment]:
    return [assess_rule(rows, stats, floor, signal) for floor, signal in product(FLOORS, SIGNAL_NAMES)]


def eligible_rules(assessments: Sequence[RuleAssessment]) -> list[RuleAssessment]:
    return [
        assessment
        for assessment in assessments
        if assessment.eligible and (assessment.floor, assessment.signal) != CURRENT_RULE
    ]


def collect_rows(index_profile: IndexProfile = "contextual-v1") -> list[FeatureRow]:
    if index_profile != "contextual-v1":
        raise RuntimeError(
            "floor sweep requires the coherent contextual-v1/off index; "
            f"received {index_profile!r}"
        )
    eval_set_integrity.verify()
    regression_set_integrity.verify()
    assert_live_index_profile(index_profile)
    retriever = build_retriever("off", expected_profile=index_profile)
    rows: list[FeatureRow] = []
    for question in eval_set_integrity.load_eval_set()["questions"]:
        rows.append(
            extract_features(
                retriever.retrieve(question["question"], k=RETRIEVE_K, top_n=CHANNEL_TOP_N),
                question_id=str(question["id"]),
                language=str(question["language"]),
                answerable=bool(question["answerable"]),
                cohort="eval_set",
                expected_chunk_ids=list(question.get("expected_chunk_ids", [])),
            )
        )
    for question in regression_set_integrity.load_regression_set()["queries"]:
        expected = [question["expected_chunk_id"]] if question.get("expected_chunk_id") else []
        rows.append(
            extract_features(
                retriever.retrieve(question["query"], k=RETRIEVE_K, top_n=CHANNEL_TOP_N),
                question_id=str(question["id"]),
                language=str(question["language"]),
                answerable=bool(question["should_answer"]),
                cohort="regression_controls",
                expected_chunk_ids=expected,
            )
        )
    return rows


def _gate_word(value: bool) -> str:
    return "PASS" if value else "FAIL"


def render_report(
    rows: Sequence[FeatureRow],
    stats: Sequence[CellStat],
    assessments: Sequence[RuleAssessment],
    manifest: index_manifest.IndexManifest,
) -> str:
    provenance = artifacts.resolve_provenance("contextual-v1", "off").render()
    lines = [
        "# Grounded-review entry-rule calibration sweep",
        "",
        f"> {_BANNER}",
        "",
        provenance,
        f"- manifest_build_commit: {manifest.build_commit}",
        "",
        "## Exhaustive candidate grid",
        "",
        "| floor | signal | G1 | G2 | G3 | G4 | mechanically eligible |",
        "|---|---|---|---|---|---|---|",
    ]
    for assessment in assessments:
        current = " (current)" if (assessment.floor, assessment.signal) == CURRENT_RULE else ""
        lines.append(
            f"| {assessment.floor:.2f}{current} | {assessment.signal} | "
            f"{_gate_word(assessment.g1_fidelity)} | "
            f"{_gate_word(assessment.g2_false_refusal_reduction)} | "
            f"{_gate_word(assessment.g3_unanswerable_containment)} | "
            f"{_gate_word(assessment.g4_regression_controls)} | "
            f"{'yes' if assessment.eligible and not current else 'no'} |"
        )
    lines += ["", "## Mechanically eligible candidates", ""]
    eligible = eligible_rules(assessments)
    lines.extend(
        [f"- floor={item.floor:.2f}, signal={item.signal}" for item in eligible]
        or ["_None._"]
    )
    lines += [
        "",
        "G5 structural preflight: PASS — this report was completed inside the configured "
        "calibration root and promoted from a `.partial` directory only after every artifact "
        "was written.",
        "",
        f"Rows evaluated: {len(rows)}; statistic cells: {len(stats)}.",
        "",
    ]
    return "\n".join(lines)


def render_sanitized_summary(
    assessments: Sequence[RuleAssessment], manifest: index_manifest.IndexManifest
) -> str:
    provenance = artifacts.resolve_provenance("contextual-v1", "off").render()
    lines = [
        "# Review-floor sweep summary",
        "",
        f"> {_BANNER}",
        "",
        provenance,
        f"- manifest_build_commit: {manifest.build_commit}",
        "",
        "## Exhaustive candidate gates",
        "",
        "| floor | signal | G1 | G2 | G3 | G4 | mechanically eligible |",
        "|---|---|---|---|---|---|---|",
    ]
    for assessment in assessments:
        current = " (current)" if (assessment.floor, assessment.signal) == CURRENT_RULE else ""
        lines.append(
            f"| {assessment.floor:.2f}{current} | {assessment.signal} | "
            f"{_gate_word(assessment.g1_fidelity)} | "
            f"{_gate_word(assessment.g2_false_refusal_reduction)} | "
            f"{_gate_word(assessment.g3_unanswerable_containment)} | "
            f"{_gate_word(assessment.g4_regression_controls)} | "
            f"{'yes' if assessment.eligible and not current else 'no'} |"
        )
    lines += [
        "",
        "## Mechanically eligible candidates",
        "",
    ]
    lines.extend(
        [f"- floor `{item.floor:.2f}`, signal `{item.signal}`" for item in eligible_rules(assessments)]
        or ["_None._"]
    )
    lines += [
        "",
        "No rule was selected or applied. A fresh confirmatory holdout, paid generation "
        "evaluation, human review, and explicit owner decisions remain required.",
        "",
    ]
    return "\n".join(lines)


def write_run(
    rows: Sequence[FeatureRow],
    manifest: index_manifest.IndexManifest,
    *,
    calibration_dir: Path,
    now: datetime | None = None,
) -> Path:
    root = calibration_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"floor_sweep_{timestamp}"
    partial = root / f"{run_name}.partial"
    final = root / run_name
    partial.mkdir(exist_ok=False)
    try:
        stats = sweep_grid(rows)
        assessments = assess_grid(rows, stats)
        (partial / "report.md").write_text(
            render_report(rows, stats, assessments, manifest), encoding="utf-8", newline="\n"
        )
        (partial / "grid.json").write_text(
            json.dumps([asdict(stat) for stat in stats], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (partial / "features.jsonl").write_text(
            "".join(
                json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ),
            encoding="utf-8",
            newline="\n",
        )
        (partial / "sanitized_summary.md").write_text(
            render_sanitized_summary(assessments, manifest), encoding="utf-8", newline="\n"
        )
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def run(*, index_profile: IndexProfile = "contextual-v1") -> Path:
    rows = collect_rows(index_profile)
    manifest = index_manifest.verify(expected_profile=index_profile)
    run_dir = write_run(rows, manifest, calibration_dir=CALIBRATION_DIR)
    assessments = assess_grid(rows, sweep_grid(rows))
    print(_BANNER)
    print(f"Sweep written to: {run_dir}")
    print(
        "Mechanically eligible candidates: "
        + str([(item.floor, item.signal) for item in eligible_rules(assessments)])
    )
    return run_dir


if __name__ == "__main__":
    run()
