from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path

import api.generation
import retrieval.hybrid
from eval import hash_eval_set, metrics

REPORT_DIR = Path(__file__).resolve().parent / "reports"

# Deliberate simplification: QueryResponse doesn't expose whether an LLM call
# actually happened (a threshold-refusal and an LLM-self-refusal both produce
# refused=True, status="ok" — indistinguishable from the response alone), so
# rather than threading that information through api/generation.py we delay
# uniformly before every question rather than only before ones that reach the
# LLM. Costs extra wall-clock (40 * 20s ~= 13 min worst case) but is simple,
# correct, and conservative against Groq's free-tier rate limits.
INTER_QUESTION_DELAY_SECONDS = 20

CSV_COLUMNS = [
    "id",
    "language",
    "question",
    "expected_answer",
    "generated_answer",
    "expected_document_id",
    "expected_section_heading",
    "cited_document_ids",
    "cited_section_headings",
    "retrieval_succeeded",
    "citation_accuracy_pass",
    "faithfulness_pass",
    "reviewer_notes",
]


def correct_refusal_rate(rows: list[dict]) -> float:
    """Fraction of unanswerable-subset rows where refused is True. 0.0 if the subset is empty."""
    unanswerable_rows = [row for row in rows if not row["answerable"]]
    if not unanswerable_rows:
        return 0.0
    return sum(1 for row in unanswerable_rows if row["refused"]) / len(unanswerable_rows)


def false_refusal_rate(rows: list[dict]) -> float:
    """Fraction of answerable-subset rows where refused is True. 0.0 if the subset is empty."""
    answerable_rows = [row for row in rows if row["answerable"]]
    if not answerable_rows:
        return 0.0
    return sum(1 for row in answerable_rows if row["refused"]) / len(answerable_rows)


def _p90(latencies_ms: list[float]) -> float:
    """Nearest-rank p90 (no interpolation) so it stays well-defined for the small n typical of this eval set."""
    if not latencies_ms:
        return 0.0
    ordered = sorted(latencies_ms)
    rank = math.ceil(0.9 * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return ordered[index]


def _build_row(question: dict) -> dict:
    time.sleep(INTER_QUESTION_DELAY_SECONDS)

    start_time = time.monotonic()
    response = api.generation.answer_question(question["question"], question["language"])
    latency_ms = (time.monotonic() - start_time) * 1000

    retrieval_succeeded = None
    if question["answerable"]:
        retrieved = retrieval.hybrid.retrieve(question["question"], k=5)
        retrieved_chunk_ids = [result.chunk_id for result in retrieved]
        retrieval_succeeded = metrics.recall_at_k(retrieved_chunk_ids, question["expected_chunk_ids"], 5) > 0

    return {
        "id": question["id"],
        "language": question["language"],
        "answerable": question["answerable"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "expected_document_id": question["expected_document_id"],
        "expected_section_heading": question["expected_section_heading"],
        "refused": response.refused,
        "status": response.status,
        "confidence": response.confidence,
        "threshold": response.threshold,
        "retrieval_succeeded": retrieval_succeeded,
        "answer": response.answer,
        "citations": response.citations,
        "latency_ms": latency_ms,
    }


def build_report(rows: list[dict], version: str) -> str:
    correct_refusal = correct_refusal_rate(rows)
    false_refusal = false_refusal_rate(rows)
    latencies = [row["latency_ms"] for row in rows]
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    p90_latency = _p90(latencies)

    lines = [
        f"# Generation Evaluation Report — eval_set v{version}",
        "",
        f"## Correct-Refusal Rate: {correct_refusal:.3f}",
        "",
        f"**False-refusal rate (answerable subset): {false_refusal:.3f}**",
        "",
        "Target thresholds from the plan, for reference only — this report does not compute a pass/fail verdict:",
        "- Correct-refusal rate target: ≥ 0.80 (8/10).",
        "- False-refusal rate target: ≤ 0.10 (≤ 3/30).",
        "",
        "## Per-Question Results",
        "",
        "| id | language | answerable (expected) | refused (actual) | status | confidence | threshold | retrieval_succeeded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        confidence = "n/a" if row["confidence"] is None else f"{row['confidence']:.4f}"
        retrieval_cell = "n/a" if row["retrieval_succeeded"] is None else str(row["retrieval_succeeded"])
        lines.append(
            f"| {row['id']} | {row['language']} | {row['answerable']} | {row['refused']} | "
            f"{row['status']} | {confidence} | {row['threshold']:.4f} | {retrieval_cell} |"
        )

    lines += [
        "",
        "## Latency Summary (all rows, n=%d)" % len(rows),
        "",
        f"- Mean latency: {mean_latency:.1f} ms",
        f"- p90 latency: {p90_latency:.1f} ms",
        "- No per-provider breakdown: `QueryResponse` does not expose which LLM provider served a request.",
        "",
        "## Manual Review Required",
        "",
        (
            "Citation accuracy and faithfulness are NOT computed by this script. Per the plan's "
            "explicit decision to grade these two headline metrics by human review rather than "
            f"LLM-as-judge, see `manual_review_checklist_v{version}.csv` for the project owner to "
            "grade by hand (answerable subset only — unanswerable-subset correctness is already "
            "fully captured by the correct-refusal-rate metric above)."
        ),
        "",
    ]

    return "\n".join(lines) + "\n"


def write_manual_review_csv(rows: list[dict], path: Path) -> None:
    answerable_rows = [row for row in rows if row["answerable"]]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in answerable_rows:
            citations = row["citations"]
            writer.writerow(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "generated_answer": row["answer"],
                    "expected_document_id": row["expected_document_id"],
                    "expected_section_heading": row["expected_section_heading"],
                    "cited_document_ids": ";".join(c.document_id for c in citations),
                    "cited_section_headings": ";".join(c.section_heading for c in citations),
                    "retrieval_succeeded": row["retrieval_succeeded"],
                    "citation_accuracy_pass": "",
                    "faithfulness_pass": "",
                    "reviewer_notes": "",
                }
            )


def run(eval_set_path: Path = hash_eval_set.EVAL_SET_FILE, report_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    hash_eval_set.verify(eval_set_path)
    data = hash_eval_set.load_eval_set(eval_set_path)
    questions = data["questions"]

    rows: list[dict] = []
    for question in questions:
        rows.append(_build_row(question))

    correct_refusal = correct_refusal_rate(rows)
    false_refusal = false_refusal_rate(rows)

    report = build_report(rows, data["version"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"generation_eval_v{data['version']}.md"
    report_path.write_text(report, encoding="utf-8")

    csv_path = report_dir / f"manual_review_checklist_v{data['version']}.csv"
    write_manual_review_csv(rows, csv_path)

    print(f"Correct-refusal rate: {correct_refusal:.3f}")
    print(f"False-refusal rate: {false_refusal:.3f}")
    print(f"Report written to: {report_path}")
    print(f"Manual review checklist written to: {csv_path}")

    return report_path, csv_path


if __name__ == "__main__":
    run()
