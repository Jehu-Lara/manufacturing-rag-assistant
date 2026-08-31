from __future__ import annotations

from pathlib import Path
from typing import Any

from src.domain.models import ExpansionMode, IndexProfile
from src.domain.policies import RefusalPolicy, top1_semantic_score_from_results
from src.domain.ports import RetrieverPort
from src.features.evaluation import artifacts, eval_set_integrity, regression_set_integrity
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.evaluation.metrics import recall_at_k
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"

# Byte-stable invariant (src/core/config.py). Diagnostic only here — this
# report never selects or ships a threshold, it just shows where each frozen
# regression query lands relative to the shipped cutoff.
DIAGNOSTIC_THRESHOLD = 0.5999


def _row(retriever: RetrieverPort, query: dict[str, Any], threshold: float) -> dict[str, Any]:
    results = retriever.retrieve(query["query"], k=SEMANTIC_EXTRACTION_K)
    ids = [r.chunk_id for r in results]
    score = top1_semantic_score_from_results(results)
    confident = RefusalPolicy(threshold).is_confident(results)
    expected_chunk_id = query["expected_chunk_id"]
    hit5 = recall_at_k(ids, [expected_chunk_id], 5) if expected_chunk_id else None
    return {
        "id": query["id"],
        "language": query["language"],
        "should_answer": query["should_answer"],
        "top1_semantic": score,
        "gate": "answer" if confident else "REFUSE",
        "recall@5": hit5,
        "top1_chunk": ids[0] if ids else None,
    }


def _fmt_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.4f}"


def run(
    expansion_mode: ExpansionMode = "off",
    *,
    index_profile: IndexProfile = "raw-v1",
    write_canonical_alias: bool = False,
) -> Path:
    if write_canonical_alias:
        artifacts.ensure_canonical_alias_allowed(index_profile, expansion_mode)

    regression_set_integrity.verify()
    eval_set_integrity.verify()
    data = regression_set_integrity.load_regression_set()
    version = eval_set_integrity.load_eval_set()["version"]
    threshold = DIAGNOSTIC_THRESHOLD

    header = artifacts.resolve_provenance(index_profile, expansion_mode)
    lines = [
        header.render(),
        "",
        f"# Regression Eval — eval_set v{version}",
        "",
        f"- threshold (diagnostic): {threshold}",
        "",
    ]
    assert_live_index_profile(index_profile)
    retriever = build_retriever(expansion_mode)
    rows = [_row(retriever, q, threshold) for q in data["queries"]]
    lines += [
        f"## expansion_mode = {expansion_mode}",
        "",
        "| id | lang | should_answer | top1_semantic | gate | recall@5 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['language']} | {r['should_answer']} | "
            f"{_fmt_score(r['top1_semantic'])} | {r['gate']} | {r['recall@5']} |"
        )
    answered_ok = sum(1 for r in rows if r["should_answer"] and r["gate"] == "answer")
    refused_ok = sum(1 for r in rows if not r["should_answer"] and r["gate"] == "REFUSE")
    n_ans = sum(1 for r in rows if r["should_answer"])
    n_ref = sum(1 for r in rows if not r["should_answer"])
    lines += [
        "",
        f"- answerable passing gate: {answered_ok}/{n_ans}",
        f"- controls correctly refused: {refused_ok}/{n_ref}",
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / artifacts.artifact_filename(
        "regression_eval", version, index_profile, expansion_mode, "md"
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if write_canonical_alias:
        (REPORT_DIR / f"regression_eval_v{version}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
    print(f"Report written to: {report}")
    return report


if __name__ == "__main__":
    _modes: tuple[ExpansionMode, ...] = ("off", "semantic", "lexical", "both")
    for _mode in _modes:
        run(_mode, index_profile="raw-v1")
