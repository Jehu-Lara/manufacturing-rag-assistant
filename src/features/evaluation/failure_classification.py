from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"

CLASSES: tuple[str, ...] = (
    "gate-over-refusal",
    "same-document-decoy",
    "cross-document-decoy",
    "retrieval-miss",
)

# Frozen ruling — the committed Phase-1 classification and the Phase-1 closeout
# (docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md §5). If a
# fresh run of the classifier over the JSONL disagrees with these, STOP and
# report the per-question breakdown — do not touch eval/eval_set.json.
EXPECTED_COUNTS: dict[str, int] = {
    "gate-over-refusal": 15,
    "same-document-decoy": 9,
    "cross-document-decoy": 2,
    "retrieval-miss": 0,
}


_DETAILS_NAME_RE = re.compile(r"__(raw-v1|contextual-v1)__(off|semantic|lexical|both)\.jsonl$")


def _profile_mode_from_details_name(name: str) -> tuple[str, str]:
    match = _DETAILS_NAME_RE.search(name)
    if match is None:
        return ("raw-v1", "off")
    return (match.group(1), match.group(2))


def _document_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def classify_failure(
    expected_chunk_ids: list[str],
    expected_document_id: str,
    top5: list[dict[str, Any]],
    gate_confident: bool,
) -> str:
    """Assumes the question is a known failure (recall@5 miss, or gate refuses a
    correctly-retrieved chunk). Outputs are mutually exclusive:

    - ``gate-over-refusal``    expected chunk IS in top-5, gate refuses;
    - ``same-document-decoy``  expected chunk absent, expected document in top-5, top-1 from it;
    - ``cross-document-decoy`` expected chunk absent, expected document in top-5, top-1 from another;
    - ``retrieval-miss``       expected document absent from top-5.
    """
    top5_chunk_ids = {entry["chunk_id"] for entry in top5}
    expected_chunk_present = any(cid in top5_chunk_ids for cid in expected_chunk_ids)
    if expected_chunk_present and not gate_confident:
        return "gate-over-refusal"

    top5_documents = {_document_of(entry["chunk_id"]) for entry in top5}
    if expected_document_id not in top5_documents:
        return "retrieval-miss"
    if top5 and _document_of(top5[0]["chunk_id"]) == expected_document_id:
        return "same-document-decoy"
    return "cross-document-decoy"


def is_failure(record: dict[str, Any]) -> bool:
    top5_chunk_ids = {entry["chunk_id"] for entry in record["top5"]}
    expected_chunk_present = any(cid in top5_chunk_ids for cid in record["expected_chunk_ids"])
    gate_confident = record["gate_decision"] == "answer"
    return not (expected_chunk_present and gate_confident)


def classify_record(record: dict[str, Any]) -> str:
    return classify_failure(
        record["expected_chunk_ids"],
        record["expected_document_id"],
        record["top5"],
        record["gate_decision"] == "answer",
    )


def load_details(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def classify_details(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    for record in records:
        if not is_failure(record):
            continue
        cls = classify_record(record)
        top1 = record["top5"][0]["chunk_id"] if record["top5"] else None
        classified.append(
            {
                "id": record["id"],
                "lang": record["lang"],
                "class": cls,
                "top1_retrieved": top1,
                "expected_chunk_ids": record["expected_chunk_ids"],
                "expected_document_id": record["expected_document_id"],
                "gate_decision": record["gate_decision"],
            }
        )
    return classified


def count_classes(classified: list[dict[str, Any]]) -> dict[str, int]:
    counts = {cls: 0 for cls in CLASSES}
    for row in classified:
        counts[row["class"]] += 1
    return counts


def build_report(
    details_path: Path,
    records: list[dict[str, Any]],
    classified: list[dict[str, Any]],
) -> str:
    counts = count_classes(classified)
    answerable_n = len(records)
    index_profile, expansion_mode = _profile_mode_from_details_name(details_path.name)
    provenance_line = (
        f"generated: {index_profile} / expansion_mode={expansion_mode} — "
        f"deterministic classifier over {details_path.name}"
    )
    decoy_rows = [r for r in classified if r["class"].endswith("decoy")]
    gate_rows = [r for r in classified if r["class"] == "gate-over-refusal"]

    lines = [
        "<!-- provenance",
        f"source: {details_path.name} (per-question top-5 dump, deterministic classifier)",
        "classifier: src/features/evaluation/failure_classification.py::classify_failure",
        provenance_line,
        "method: for each answerable question, if it is a failure (recall@5 miss OR gate refuses a",
        "  correctly-retrieved chunk), classify into one mutually-exclusive class:",
        "  gate-over-refusal    = expected chunk IS in top-5 but the 0.5999 gate refuses",
        "  same-document-decoy  = expected chunk NOT in top-5, expected document IS, top-1 from it",
        "  cross-document-decoy = expected chunk NOT in top-5, expected document IS, top-1 from another",
        "  retrieval-miss       = expected document absent from top-5",
        "-->",
        "",
        f"# Failure classification — eval_set v1.1.0, {index_profile} index, expansion_mode={expansion_mode}",
        "",
        "Reproducible from committed machine-readable evidence: regenerate with",
        "`python -m src.features.evaluation.failure_classification`.",
        "",
        f"## Counts (answerable subset, n={answerable_n})",
        "",
        "| class | count |",
        "|---|---|",
        f"| gate-over-refusal (retrieval OK, gate refuses) | {counts['gate-over-refusal']} |",
        f"| same-document-decoy | {counts['same-document-decoy']} |",
        f"| cross-document-decoy | {counts['cross-document-decoy']} |",
        f"| retrieval-miss (expected document absent from top-5) | **{counts['retrieval-miss']}** |",
        "",
        (
            f"**Non-gate retrieval failures = {counts['same-document-decoy'] + counts['cross-document-decoy']} "
            f"({counts['same-document-decoy']} same-document + {counts['cross-document-decoy']} cross-document). "
            f"{counts['retrieval-miss']} true retrieval misses.**"
        ),
        "",
        "## Decoy failures (expected chunk absent, expected document still in top-5)",
        "",
        "| id | lang | class | top-1 retrieved | expected |",
        "|---|---|---|---|---|",
    ]
    for row in decoy_rows:
        expected = ", ".join(row["expected_chunk_ids"])
        lines.append(
            f"| {row['id']} | {row['lang']} | {row['class']} | `{row['top1_retrieved']}` | `{expected}` |"
        )

    lines += [
        "",
        "## Gate-over-refusals (expected chunk retrieved, 0.5999 gate refuses)",
        "",
        "| id | lang | top-1 retrieved | expected |",
        "|---|---|---|---|",
    ]
    for row in gate_rows:
        expected = ", ".join(row["expected_chunk_ids"])
        lines.append(
            f"| {row['id']} | {row['lang']} | `{row['top1_retrieved']}` | `{expected}` |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def run(
    details_path: Path | None = None,
    out_path: Path | None = None,
) -> tuple[Path, dict[str, int]]:
    details_path = details_path or REPORT_DIR / "retrieval_details_v1.1.0__raw-v1__off.jsonl"
    out_path = out_path or REPORT_DIR / "classification_v1.1.0__raw-v1__off.md"

    records = load_details(details_path)
    classified = classify_details(records)
    counts = count_classes(classified)

    report = build_report(details_path, records, classified)
    out_path.write_text(report, encoding="utf-8", newline="\n")

    print(f"Classification written to: {out_path}")
    for cls in CLASSES:
        marker = "" if counts[cls] == EXPECTED_COUNTS[cls] else "  <-- MISMATCH"
        print(f"  {cls}: {counts[cls]} (expected {EXPECTED_COUNTS[cls]}){marker}")
    if counts != EXPECTED_COUNTS:
        print("COUNTS DIVERGE FROM THE FROZEN RULING — do not edit eval_set.json; report the breakdown.")
    return out_path, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    _, counts = run(args.details, args.out)
    raise SystemExit(0 if counts == EXPECTED_COUNTS else 1)


if __name__ == "__main__":
    main()
