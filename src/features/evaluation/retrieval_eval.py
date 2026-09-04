from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME
from src.core.config import DEFAULT_REFUSAL_COSINE_THRESHOLD
from src.core.paths import EVAL_REPORTS_DIR, REPO_ROOT
from src.domain.models import ExpansionMode, IndexProfile, RetrievalResult
from src.domain.policies import RefusalPolicy
from src.features.evaluation import artifacts, eval_set_integrity, metrics
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K, HybridRetriever

REPORT_DIR = EVAL_REPORTS_DIR


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _score_answerable(retriever: HybridRetriever, question: dict[str, Any]) -> dict[str, Any]:
    results = retriever.retrieve(question["question"], k=SEMANTIC_EXTRACTION_K)
    retrieved_ids = [r.chunk_id for r in results]
    return {
        "id": question["id"],
        "language": question["language"],
        "retrieved": results,
        "gate_confident": RefusalPolicy(DEFAULT_REFUSAL_COSINE_THRESHOLD).is_confident(results),
        "recall@3": metrics.recall_at_k(retrieved_ids, question["expected_chunk_ids"], 3),
        "recall@5": metrics.recall_at_k(retrieved_ids, question["expected_chunk_ids"], 5),
        # Scoped to the original top-5 window (retrieve() used to be called with
        # k=5 directly) so widening k above to capture the semantic top-1 doesn't
        # change this metric's historical meaning.
        "rr": metrics.reciprocal_rank(retrieved_ids[:5], question["expected_chunk_ids"]),
        "top1_semantic_score": metrics.top1_semantic_score(results),
    }


def _retrieval_detail_record(question: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable per-question top-5 dump. Absent semantic/BM25 ranks and
    scores are JSON `null`, never invented zeros. No query text, generated
    answers, prompts, or secrets — only what a deterministic failure classifier
    needs."""
    top5: list[dict[str, Any]] = []
    for rank, result in enumerate(row["retrieved"][:5], start=1):
        top5.append(
            {
                "chunk_id": result.chunk_id,
                "rank": rank,
                "semantic_score": result.semantic_score,
                "semantic_rank": result.semantic_rank,
                "bm25_rank": result.bm25_rank,
                "fused_score": result.fused_score,
            }
        )
    return {
        "id": question["id"],
        "lang": question["language"],
        "top5": top5,
        "gate_decision": "answer" if row["gate_confident"] else "refuse",
        "expected_document_id": question["expected_document_id"],
        "expected_chunk_ids": question["expected_chunk_ids"],
    }


def _write_retrieval_details(
    path: Path, answerable: list[dict[str, Any]], answerable_rows: list[dict[str, Any]]
) -> None:
    row_by_id = {row["id"]: row for row in answerable_rows}
    payload = "\n".join(
        json.dumps(_retrieval_detail_record(q, row_by_id[q["id"]]), ensure_ascii=False)
        for q in answerable
    )
    tmp_path = path.parent / (path.name + ".tmp")
    tmp_path.write_text(payload + "\n", encoding="utf-8", newline="\n")
    tmp_path.replace(path)


def _score_unanswerable(retriever: HybridRetriever, question: dict[str, Any]) -> dict[str, Any]:
    results = retriever.retrieve(question["question"], k=SEMANTIC_EXTRACTION_K)
    return {
        "id": question["id"],
        "top1_fused_score": results[0].fused_score if results else 0.0,
        "top1_semantic_score": metrics.top1_semantic_score(results),
        "retrieved": results,
    }


def _format_result_line(result: RetrievalResult) -> str:
    return (
        f"  - `{result.chunk_id}` (fused={result.fused_score:.4f}, "
        f"semantic_rank={result.semantic_rank}, bm25_rank={result.bm25_rank}) — "
        f"{result.metadata.get('document_title', '')} / {result.metadata.get('section_heading', '')}"
    )


def _id_suffix_number(question_id: str) -> int:
    digits = "".join(ch for ch in question_id if ch.isdigit())
    return int(digits) if digits else 0


def _matched_pair_gap_section(
    questions: list[dict[str, Any]],
    answerable_rows: list[dict[str, Any]],
) -> list[str]:
    """en - es top-1 semantic-score gap for answerable questions paired by
    set-equal `expected_chunk_ids` (ties broken by nearest id suffix). This is
    the frozen matched-pair signal the per-language threshold discussion is
    gated on — see the Phase 3 bilingual-refusal design doc."""
    expected_by_id = {q["id"]: set(q["expected_chunk_ids"]) for q in questions}
    score_by_id = {row["id"]: row["top1_semantic_score"] for row in answerable_rows}
    en_rows = [row for row in answerable_rows if row["language"] == "en"]
    es_rows = [row for row in answerable_rows if row["language"] == "es"]

    pairs: list[dict[str, Any]] = []
    used_es_ids: set[str] = set()
    for en_row in en_rows:
        en_expected = expected_by_id.get(en_row["id"], set())
        candidates = [
            es_row
            for es_row in es_rows
            if es_row["id"] not in used_es_ids and expected_by_id.get(es_row["id"], set()) == en_expected
        ]
        if not candidates:
            continue
        es_row = min(
            candidates,
            key=lambda c: abs(_id_suffix_number(c["id"]) - _id_suffix_number(en_row["id"])),
        )
        used_es_ids.add(es_row["id"])
        pairs.append(
            {
                "en_id": en_row["id"],
                "es_id": es_row["id"],
                "en_score": score_by_id[en_row["id"]],
                "es_score": score_by_id[es_row["id"]],
                "gap": score_by_id[en_row["id"]] - score_by_id[es_row["id"]],
            }
        )

    lines = ["## Matched-pair cosine gap (en − es)", ""]
    if not pairs:
        lines += ["_No en/es answerable pairs share `expected_chunk_ids`._", ""]
        return lines

    lines += [
        (
            "Answerable en/es questions paired by set-equal `expected_chunk_ids` "
            "(ties broken by nearest id). Gap = en top-1 semantic score − es top-1 semantic score."
        ),
        "",
        "| en id | es id | en top-1 semantic | es top-1 semantic | gap (en − es) |",
        "|---|---|---|---|---|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['en_id']} | {pair['es_id']} | {pair['en_score']:.4f} | "
            f"{pair['es_score']:.4f} | {pair['gap']:+.4f} |"
        )
    mean_gap = sum(pair["gap"] for pair in pairs) / len(pairs)
    lines += ["", f"- **Mean gap (en − es), n={len(pairs)}**: {mean_gap:+.4f}", ""]
    return lines


def build_report(
    questions: list[dict[str, Any]],
    answerable_rows: list[dict[str, Any]],
    unanswerable_rows: list[dict[str, Any]],
    version: str,
) -> str:
    recall_at_3 = sum(row["recall@3"] for row in answerable_rows) / len(answerable_rows)
    recall_at_5 = sum(row["recall@5"] for row in answerable_rows) / len(answerable_rows)
    mrr = metrics.mean_reciprocal_rank([row["rr"] for row in answerable_rows])

    by_language: dict[str, list[dict[str, Any]]] = {}
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
        *_matched_pair_gap_section(questions, answerable_rows),
        "## Unanswerable Subset (n=%d) — Top-1 Score Distribution" % len(unanswerable_rows),
        "",
        f"- Fused score: {unanswerable_fused_summary}",
        f"- Semantic score (pure cosine similarity, `semantic_rank == 1`): {unanswerable_semantic_summary}",
        (
            "- No refusal/gating decision is made here — a separate threshold analysis "
            "picks a threshold from raw semantic_score (see "
            f"`eval/reports/threshold_analysis_v{version}.md`), since fused_score "
            "is rank-based and disqualified as a refusal-confidence signal."
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
    version = data["version"]
    questions = data["questions"]

    assert_live_index_profile(index_profile)
    retriever = build_retriever(expansion_mode, expected_profile=index_profile)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    answerable_rows = [_score_answerable(retriever, q) for q in answerable]
    unanswerable_rows = [_score_unanswerable(retriever, q) for q in unanswerable]

    header = artifacts.resolve_provenance(index_profile, expansion_mode)
    report = header.render() + "\n\n" + build_report(
        questions, answerable_rows, unanswerable_rows, version
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / artifacts.artifact_filename(
        "retrieval_report", version, index_profile, expansion_mode, "md"
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")

    details_path = REPORT_DIR / artifacts.artifact_filename(
        "retrieval_details", version, index_profile, expansion_mode, "jsonl"
    )
    _write_retrieval_details(details_path, answerable, answerable_rows)

    if write_canonical_alias:
        (REPORT_DIR / f"retrieval_report_v{version}.md").write_text(
            report, encoding="utf-8", newline="\n"
        )

    recall_at_5 = sum(row["recall@5"] for row in answerable_rows) / len(answerable_rows)
    print(f"Recall@3: {sum(row['recall@3'] for row in answerable_rows) / len(answerable_rows):.3f}")
    print(f"Recall@5: {recall_at_5:.3f}")
    print(f"MRR: {metrics.mean_reciprocal_rank([row['rr'] for row in answerable_rows]):.3f}")
    print(f"Report written to: {report_path}")
    print(f"Retrieval details written to: {details_path}")

    return report_path


if __name__ == "__main__":
    run()
