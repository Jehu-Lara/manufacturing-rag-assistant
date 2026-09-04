from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.features.evaluation.gate_eval.models import (
    _POLICIES,
    GateResult,
    QuestionOutcome,
    RetrievalSnapshot,
)

_CHECKLIST_HEADER = [
    "row_id",
    "arm",
    "repeat",
    "question_id",
    "language",
    "answerable",
    "refused",
    "answer",
    "cited_chunk_ids",
    "cited_chunk_texts",
    "expected_answer",
    "expected_chunk_ids",
    "citation_accuracy_pass",
    "faithfulness_pass",
    "safe_pass",
    "notes",
]
_EDITABLE_COLUMNS = ("citation_accuracy_pass", "faithfulness_pass", "safe_pass", "notes")
_VERDICT_COLUMNS = ("citation_accuracy_pass", "faithfulness_pass", "safe_pass")
_IMMUTABLE_COLUMNS = tuple(c for c in _CHECKLIST_HEADER if c not in _EDITABLE_COLUMNS)


def _arm_labels(run_id: str) -> dict[str, str]:
    """Deterministic but not guessable: order the two policies by a hash of
    (run_id, policy). Grader sees `arm-A` / `arm-B`, the mapping lives only in
    arm_map.sealed.json."""
    ordered = sorted(_POLICIES, key=lambda p: hashlib.sha256(f"{run_id}|{p}".encode()).hexdigest())
    return {ordered[0]: "arm-A", ordered[1]: "arm-B"}


def _checklist_rows(
    run_id: str,
    outcomes: list[QuestionOutcome],
    chunk_text: dict[str, str],
    expected_answers: dict[str, str],
) -> list[dict[str, str]]:
    arm = _arm_labels(run_id)
    rows: list[dict[str, str]] = []
    for outcome in outcomes:
        needs_grading = (outcome.answerable and not outcome.refused and outcome.status == "ok") or (
            outcome.is_unsafe_unanswerable
        )
        if not needs_grading:
            continue
        rows.append(
            {
                "row_id": f"{arm[outcome.policy]}-r{outcome.repeat}-{outcome.question_id}",
                "arm": arm[outcome.policy],
                "repeat": str(outcome.repeat),
                "question_id": outcome.question_id,
                "language": outcome.language,
                "answerable": str(outcome.answerable),
                "refused": str(outcome.refused),
                "answer": outcome.answer_text,
                "cited_chunk_ids": ";".join(outcome.cited_chunk_ids),
                "cited_chunk_texts": " || ".join(
                    chunk_text.get(cid, "<not in snapshot>") for cid in outcome.cited_chunk_ids
                ),
                "expected_answer": expected_answers.get(outcome.question_id, ""),
                "expected_chunk_ids": ";".join(outcome.expected_chunk_ids),
                "citation_accuracy_pass": "",
                "faithfulness_pass": "",
                "safe_pass": "",
                "notes": "",
            }
        )
    return rows


def _write_checklist(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CHECKLIST_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def checklist_baseline(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """The immutable half of every checklist row, keyed by row_id. Sealed at run
    time; on import the graded CSV must reproduce it exactly - a deleted,
    added, duplicated or altered row is rejected before any gate is scored."""
    return {row["row_id"]: {col: row[col] for col in _IMMUTABLE_COLUMNS} for row in rows}


# --------------------------------------------------------------------------- #
# Reporting + artifacts                                                       #
# --------------------------------------------------------------------------- #


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def render_comparison(
    manifest: dict[str, Any],
    snapshots: list[RetrievalSnapshot],
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    gates: list[GateResult],
) -> str:
    lines = [
        "# Gate Generation Eval - binary vs grounded_review",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Index: `{manifest['index_profile']}` / `{manifest['expansion_mode']}`  build `{manifest['build_commit']}`",
        f"- Floor/threshold: {manifest['review_floor']} / {manifest['threshold']}",
        f"- Provider: `{manifest['llm_provider']}` (provider fallback disabled)",
        f"- Repeats: holdout x{manifest['full_repeats']}, canary x{manifest['canary_repeats']}",
        f"- Verdicts imported: {manifest.get('verdicts_imported', False)}",
        "",
        "## Gate verdicts",
        "",
        "| gate | verdict | detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.detail} |")
    all_pass = all(g.passed for g in gates)
    lines += [
        "",
        f"**Automated gates: {'ALL PASS' if all_pass else 'NOT ALL PASS'}** - the ship decision, the "
        "default flip and ADR-009 Accepted are the owner's, not this runner's.",
        "",
        "## Call + cost accounting (holdout)",
        "",
        "| policy | logical | forwarded | physical attempts | failed | rate-limited | repaired | tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy in _POLICIES:
        rows = [r for r in holdout if r.policy == policy]
        lines.append(
            f"| {policy} | {sum(r.logical_calls for r in rows)} | {sum(r.forwarded_calls for r in rows)} "
            f"| {sum(r.physical_attempts for r in rows)} | {sum(r.physical_failed for r in rows)} "
            f"| {sum(r.rate_limited for r in rows)} | {sum(r.repaired for r in rows)} "
            f"| {sum(r.total_tokens for r in rows)} |"
        )
    gen = _percentiles([ms for r in holdout for ms in r.llm_latencies_ms])
    retr = _percentiles(manifest.get("retrieval_latencies_ms", []))
    lines += [
        "",
        "## Latency (modelled, not question wall-clock)",
        "",
        f"- Generation, per physical LLM call: p50 {gen['p50']:.0f} ms, p95 {gen['p95']:.0f} ms",
        f"- Retrieval, per query on the live index: p50 {retr['p50']:.0f} ms, p95 {retr['p95']:.0f} ms",
        "- A production grey-band request ~= one retrieval + one generation. Confident-band reuse "
        "and replayed retrieval are excluded from these figures by construction.",
        "",
        "## Retrieval snapshot band distribution",
        "",
        "| band | count |",
        "|---|---|",
    ]
    band_counts: dict[str, int] = {}
    for snap in snapshots:
        band_counts[snap.gate_band] = band_counts.get(snap.gate_band, 0) + 1
    for band, count in sorted(band_counts.items()):
        lines.append(f"| {band} | {count} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _finalize_dir(partial: Path, final: Path) -> None:
    checksums = "\n".join(
        f"{_sha256_file(p)}  {p.name}"
        for p in sorted(partial.iterdir())
        if p.name != "checksums.txt" and p.is_file()
    )
    (partial / "checksums.txt").write_text(checksums + "\n", encoding="utf-8")
    partial.rename(final)


def write_run_dir(
    out_root: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    snapshots: list[RetrievalSnapshot],
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    gates: list[GateResult],
    checklist_rows: list[dict[str, str]],
    arm_map: dict[str, str],
) -> Path:
    final = out_root / run_id
    if final.exists():
        raise FileExistsError(f"{final} already exists - runs are write-once, never overwritten")
    partial = out_root / f"{run_id}.partial"
    if partial.exists():
        raise FileExistsError(f"{partial} left by an aborted run - inspect and remove it by hand")
    partial.mkdir(parents=True, exist_ok=False)

    (partial / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(partial / "retrieval.jsonl", [asdict(s) for s in snapshots])
    _write_jsonl(partial / "outcomes.jsonl", [asdict(o) for o in holdout + canary])
    (partial / "comparison.md").write_text(
        render_comparison(manifest, snapshots, holdout, canary, gates), encoding="utf-8"
    )
    _write_checklist(partial / "blind_checklist.csv", checklist_rows)
    (partial / "blind_checklist.baseline.json").write_text(
        json.dumps(checklist_baseline(checklist_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (partial / "arm_map.sealed.json").write_text(
        json.dumps(arm_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _finalize_dir(partial, final)
    return final
