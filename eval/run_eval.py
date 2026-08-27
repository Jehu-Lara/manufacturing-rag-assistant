from __future__ import annotations

import subprocess
from pathlib import Path

from eval import hash_eval_set, metrics
from retrieval.embedder import MODEL_NAME
from retrieval.hybrid import SEMANTIC_EXTRACTION_K, retrieve

REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _score_answerable(question: dict) -> dict:
    results = retrieve(question["question"], k=SEMANTIC_EXTRACTION_K)
    retrieved_ids = [r.chunk_id for r in results]
    return {
        "id": question["id"],
        "language": question["language"],
        "retrieved": results,
        "recall@3": metrics.recall_at_k(retrieved_ids, question["expected_chunk_ids"], 3),
        "recall@5": metrics.recall_at_k(retrieved_ids, question["expected_chunk_ids"], 5),
        # Scoped to the original top-5 window (retrieve() used to be called with
        # k=5 directly) so widening k above to capture the semantic top-1 doesn't
        # change this metric's historical meaning.
        "rr": metrics.reciprocal_rank(retrieved_ids[:5], question["expected_chunk_ids"]),
        "top1_semantic_score": metrics.top1_semantic_score(results),
    }


def _score_unanswerable(question: dict) -> dict:
    results = retrieve(question["question"], k=SEMANTIC_EXTRACTION_K)
    return {
        "id": question["id"],
        "top1_fused_score": results[0].fused_score if results else 0.0,
        "top1_semantic_score": metrics.top1_semantic_score(results),
        "retrieved": results,
    }


def _format_result_line(result) -> str:
    return (
        f"  - `{result.chunk_id}` (fused={result.fused_score:.4f}, "
        f"semantic_rank={result.semantic_rank}, bm25_rank={result.bm25_rank}) — "
        f"{result.metadata.get('document_title', '')} / {result.metadata.get('section_heading', '')}"
    )


def build_report(
    questions: list[dict], answerable_rows: list[dict], unanswerable_rows: list[dict], version: str
) -> str:
    recall_at_3 = sum(row["recall@3"] for row in answerable_rows) / len(answerable_rows)
    recall_at_5 = sum(row["recall@5"] for row in answerable_rows) / len(answerable_rows)
    mrr = metrics.mean_reciprocal_rank([row["rr"] for row in answerable_rows])

    by_language: dict[str, list[dict]] = {}
    for row in answerable_rows:
        by_language.setdefault(row["language"], []).append(row)
    language_lines = [
        f"- **{lang}** (n={len(rows)}): recall@5 = {sum(r['recall@5'] for r in rows) / len(rows):.3f}"
        for lang, rows in sorted(by_language.items())
    ]

    unanswerable_fused_scores = [row["top1_fused_score"] for row in unanswerable_rows]
    unanswerable_fused_summary = (
        f"min={min(unanswerable_fused_scores):.4f}, max={max(unanswerable_fused_scores):.4f}, "
        f"mean={sum(unanswerable_fused_scores) / len(unanswerable_fused_scores):.4f}"
    )
    unanswerable_semantic_scores = [row["top1_semantic_score"] for row in unanswerable_rows]
    unanswerable_semantic_summary = (
        f"min={min(unanswerable_semantic_scores):.4f}, max={max(unanswerable_semantic_scores):.4f}, "
        f"mean={sum(unanswerable_semantic_scores) / len(unanswerable_semantic_scores):.4f}"
    )

    lines = [
        f"# Retrieval Evaluation Report — eval_set v{version}",
        "",
        f"- Embedding model: `{MODEL_NAME}`",
        "- Fusion: Reciprocal Rank Fusion (k=60)",
        f"- Git commit: `{_git_commit_hash()}`",
        "- Eval set SHA-256: verified against stored hash before running",
        "",
        "## Summary Metrics (answerable subset, n=%d)" % len(answerable_rows),
        "",
        f"- **Recall@3**: {recall_at_3:.3f}",
        f"- **Recall@5**: {recall_at_5:.3f}",
        f"- **MRR**: {mrr:.3f}",
        "",
        "### Recall@5 by query language",
        "",
        *language_lines,
        "",
        "## Unanswerable Subset (n=%d) — Top-1 Score Distribution" % len(unanswerable_rows),
        "",
        f"- Fused score: {unanswerable_fused_summary}",
        f"- Semantic score (pure cosine similarity, `semantic_rank == 1`): {unanswerable_semantic_summary}",
        (
            "- No refusal/gating decision is made here — Phase 3 picks a threshold from "
            "raw semantic_score (see `eval/reports/threshold_analysis_v%s.md`), since fused_score "
            "is rank-based and disqualified as a refusal-confidence signal — see SPEC.md's "
            "Phase 2 status." % version
        ),
        "",
        "## Per-Question Results (answerable subset)",
        "",
        "| id | language | recall@3 | recall@5 | RR | top-1 semantic score |",
        "|---|---|---|---|---|---|",
    ]
    for row in answerable_rows:
        lines.append(
            f"| {row['id']} | {row['language']} | {row['recall@3']:.2f} | {row['recall@5']:.2f} | "
            f"{row['rr']:.2f} | {row['top1_semantic_score']:.4f} |"
        )

    lines += [
        "",
        "## Per-Question Results (unanswerable subset)",
        "",
        "| id | top-1 fused score | top-1 semantic score |",
        "|---|---|---|",
    ]
    for row in unanswerable_rows:
        lines.append(f"| {row['id']} | {row['top1_fused_score']:.4f} | {row['top1_semantic_score']:.4f} |")

    lines += ["", "## Example Queries", ""]
    examples = answerable_rows[:2] + unanswerable_rows[:1]
    question_by_id = {q["id"]: q for q in questions}
    for row in examples:
        question = question_by_id[row["id"]]
        lines.append(f"### {row['id']} ({question['language']}): {question['question']}")
        lines.append("")
        lines.append("Top results:")
        for result in row["retrieved"][:3]:
            lines.append(_format_result_line(result))
        lines.append("")

    return "\n".join(lines) + "\n"


def run() -> Path:
    hash_eval_set.verify()
    data = hash_eval_set.load_eval_set()
    questions = data["questions"]

    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    answerable_rows = [_score_answerable(q) for q in answerable]
    unanswerable_rows = [_score_unanswerable(q) for q in unanswerable]

    report = build_report(questions, answerable_rows, unanswerable_rows, data["version"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"retrieval_report_v{data['version']}.md"
    report_path.write_text(report, encoding="utf-8")

    recall_at_5 = sum(row["recall@5"] for row in answerable_rows) / len(answerable_rows)
    print(f"Recall@3: {sum(row['recall@3'] for row in answerable_rows) / len(answerable_rows):.3f}")
    print(f"Recall@5: {recall_at_5:.3f}")
    print(f"MRR: {metrics.mean_reciprocal_rank([row['rr'] for row in answerable_rows]):.3f}")
    print(f"Report written to: {report_path}")

    return report_path


if __name__ == "__main__":
    run()
