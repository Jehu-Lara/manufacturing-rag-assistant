from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, cast

from src.domain.models import ExpansionMode, IndexProfile
from src.domain.policies import top1_semantic_score_from_results
from src.domain.ports import RetrieverPort
from src.features.evaluation import gate_holdout_integrity
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.evaluation.gate_score_guard import GATE_CONFIDENT_THRESHOLD, GATE_REVIEW_FLOOR
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"

# Before the holdout is frozen it must actually exercise the grounded-review
# band: each EN/ES x answerable/unanswerable cell needs at least this many
# questions whose top1-semantic lands in [floor, threshold). A holdout that
# barely touches the grey band cannot tell us whether grounded_review helps or
# hurts, and a paid generation run over it would be wasted spend.
MIN_GREY_PER_CELL = 3

_BANDS = ("hard_refuse", "grounded_review", "confident")


def _band_for(score: Optional[float]) -> str:
    if score is None or score < GATE_REVIEW_FLOOR:
        return "hard_refuse"
    if score < GATE_CONFIDENT_THRESHOLD:
        return "grounded_review"
    return "confident"


@dataclass(frozen=True)
class CellProfile:
    language: str
    answerable: bool
    total: int
    hard_refuse: int
    grounded_review: int
    confident: int
    grey_ids: list[str]

    @property
    def meets_minimum(self) -> bool:
        return self.grounded_review >= MIN_GREY_PER_CELL


def profile_holdout(
    *,
    holdout_path: Path = gate_holdout_integrity.GATE_HOLDOUT_FILE,
    index_profile: IndexProfile = "contextual-v1",
    expansion_mode: ExpansionMode = "off",
    retriever: Optional[RetrieverPort] = None,
) -> tuple[str, list[CellProfile]]:
    data = gate_holdout_integrity.load_gate_holdout(holdout_path)
    status = str(data.get("status", "<none>"))
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != gate_holdout_integrity.REQUIRED_TOTAL:
        raise ValueError(
            f"gate holdout must have {gate_holdout_integrity.REQUIRED_TOTAL} questions before it can "
            f"be profiled, got {len(questions) if isinstance(questions, list) else 'non-list'}"
        )

    if retriever is None:
        assert_live_index_profile(index_profile)
        retriever = build_retriever(expansion_mode)

    buckets: dict[tuple[str, bool], dict[str, list[str]]] = {
        (lang, answerable): {band: [] for band in _BANDS}
        for lang in ("en", "es")
        for answerable in (True, False)
    }
    for question in questions:
        results = retriever.retrieve(question["question"], k=SEMANTIC_EXTRACTION_K)
        band = _band_for(top1_semantic_score_from_results(results))
        buckets[(question["language"], bool(question["answerable"]))][band].append(str(question["id"]))

    cells = [
        CellProfile(
            language=lang,
            answerable=answerable,
            total=sum(len(ids) for ids in bands.values()),
            hard_refuse=len(bands["hard_refuse"]),
            grounded_review=len(bands["grounded_review"]),
            confident=len(bands["confident"]),
            grey_ids=sorted(bands["grounded_review"]),
        )
        for (lang, answerable), bands in buckets.items()
    ]
    return status, cells


def render_report(status: str, cells: list[CellProfile]) -> str:
    lines = [
        "# Gate Holdout Band Profile",
        "",
        f"- Holdout status: `{status}`",
        f"- Band: `[{GATE_REVIEW_FLOOR}, {GATE_CONFIDENT_THRESHOLD})` on contextual-v1/off",
        f"- Required grounded-review questions per cell: {MIN_GREY_PER_CELL}",
        "",
        "| language | class | total | hard_refuse | grounded_review | confident | meets min |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in sorted(cells, key=lambda c: (c.language, not c.answerable)):
        klass = "answerable" if cell.answerable else "unanswerable"
        lines.append(
            f"| {cell.language} | {klass} | {cell.total} | {cell.hard_refuse} | "
            f"{cell.grounded_review} | {cell.confident} | {'yes' if cell.meets_minimum else 'NO'} |"
        )
    lines += ["", "## Grounded-review question ids per cell", ""]
    for cell in sorted(cells, key=lambda c: (c.language, not c.answerable)):
        klass = "answerable" if cell.answerable else "unanswerable"
        lines.append(f"- {cell.language}/{klass}: {cell.grey_ids or '(none)'}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(
    *,
    holdout_path: Path = gate_holdout_integrity.GATE_HOLDOUT_FILE,
    index_profile: IndexProfile = "contextual-v1",
    expansion_mode: ExpansionMode = "off",
    retriever: Optional[RetrieverPort] = None,
    report_dir: Path = REPORT_DIR,
) -> Path:
    status, cells = profile_holdout(
        holdout_path=holdout_path,
        index_profile=index_profile,
        expansion_mode=expansion_mode,
        retriever=retriever,
    )
    report = render_report(status, cells)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "gate_holdout_band_profile.md"
    report_path.write_text(report, encoding="utf-8")
    (report_dir / "gate_holdout_band_profile.json").write_text(
        json.dumps({"status": status, "cells": [asdict(c) for c in cells]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Report written to: {report_path}")

    short = [
        f"{c.language}/{'answerable' if c.answerable else 'unanswerable'} has {c.grounded_review}"
        for c in cells
        if not c.meets_minimum
    ]
    if short:
        print(
            "GATE HOLDOUT PROFILE FAILED: cells below the "
            f"{MIN_GREY_PER_CELL}-question grounded-review minimum: " + "; ".join(short),
            file=sys.stderr,
        )
        print(
            "  Author another draft with more borderline questions in those cells "
            "BEFORE spending on a paid run. Do not lower the floor to manufacture grey cases.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("gate holdout profile OK - every cell exercises the grounded-review band")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-profile", default="contextual-v1", choices=["raw-v1", "contextual-v1"])
    parser.add_argument("--expansion-mode", default="off", choices=["off", "semantic", "lexical", "both"])
    args = parser.parse_args()
    run(
        index_profile=cast("IndexProfile", args.index_profile),
        expansion_mode=cast("ExpansionMode", args.expansion_mode),
    )


if __name__ == "__main__":
    main()
