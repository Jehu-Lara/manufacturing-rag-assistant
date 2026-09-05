"""Fusion-parameter sweep — the follow-up the ablation asked for, default-off.

`ablation_eval` established that the lexical channel is net-negative on this
set: zero exclusive rescues, and eight questions the semantic channel alone
retrieves get demoted out of the top 5 by fusion. It named RRF's rank-only
scoring as the confounder to rule out before concluding anything about BM25
itself. This module rules it in or out with numbers.

The mechanism under test, stated so the report can be read against it. RRF with
`k=60` over 20 candidates scores rank 1 at 1/61 and rank 20 at 1/80 — a 1.31x
spread across the whole list. A chunk found by BOTH channels scores the sum of
two such terms, roughly 2x. So membership in both rankings is worth more than
any rank difference within one ranking: a mediocre chunk at semantic #11 that
BM25 also liked outranks the gold chunk at semantic #1 that BM25 missed. That
is a property of the fusion parameters, not of BM25's retrieval quality, and
the two have very different fixes.

Fusion is a pure function of the two ranked id lists, so this replays it
offline over channel outputs captured once per question. That is exact, not an
approximation — `test_baseline_rule_reproduces_the_shipped_retriever_exactly`
pins the baseline row against the real `HybridRetriever`.

Default-off by construction: no index writes, no LLM calls, no served default
changed. RRF `k=60` is a byte-stable invariant (CLAUDE.md); this module reads
it and never writes it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from src.core.paths import EVAL_REPORTS_DIR
from src.domain.models import IndexProfile
from src.domain.policies import RRF_K, fuse_rankings, rrf_scores
from src.features.evaluation import ablation_eval, artifacts, eval_set_integrity, metrics
from src.features.evaluation._eval_retriever import build_retriever
from src.features.retrieval.use_cases import DEFAULT_TOP_N

FusionRule = Callable[[Sequence[str], Sequence[str]], list[tuple[str, float]]]

BASELINE_RULE = "rrf_k60"

_RECALL_K = 5


@dataclass(frozen=True)
class QuestionChannels:
    """One question's raw channel outputs, captured once and reused by every
    rule — so the sweep varies the fusion and nothing else."""

    qid: str
    language: str
    expected_chunk_ids: list[str]
    semantic_ids: list[str]
    lexical_ids: list[str]


def weighted_rrf(
    semantic_ranked_ids: Sequence[str],
    lexical_ranked_ids: Sequence[str],
    *,
    k: int,
    semantic_weight: float,
    lexical_weight: float,
) -> list[tuple[str, float]]:
    """Generalises `policies.fuse_rankings`; at k=RRF_K and unit weights it IS
    `fuse_rankings`, which is what makes a weight sweep a one-variable change.

    A zero weight drops that channel's exclusive candidates outright rather than
    scoring them 0.0 — at 0.0 the ascending-chunk_id tie-break would still let
    them surface above nothing, which is not what 'this channel is off' means.
    """
    semantic = rrf_scores(semantic_ranked_ids, k)
    lexical = rrf_scores(lexical_ranked_ids, k)
    candidates = set(semantic_ranked_ids if semantic_weight else ()) | set(
        lexical_ranked_ids if lexical_weight else ()
    )
    fused = [
        (cid, semantic_weight * semantic.get(cid, 0.0) + lexical_weight * lexical.get(cid, 0.0))
        for cid in candidates
    ]
    fused.sort(key=lambda pair: (-pair[1], pair[0]))
    return fused


def _rrf_at(k: int) -> FusionRule:
    return lambda semantic, lexical: fuse_rankings(semantic, lexical, k)


def _weighted_at(k: int, semantic_weight: float, lexical_weight: float = 1.0) -> FusionRule:
    return lambda semantic, lexical: weighted_rrf(
        semantic, lexical, k=k, semantic_weight=semantic_weight, lexical_weight=lexical_weight
    )


# Two levers, swept separately so the report can attribute any movement.
#   k  — how flat the within-channel rank curve is. Smaller k spreads ranks 1..20
#        further apart, so a rank difference can outweigh a both-channels bonus.
#   weight — how much each channel's agreement is worth at all.
# `semantic_only` is the ablation's arm expressed as a fusion rule; it must
# reproduce that report's number, which is a free cross-check on both modules.
RULES: dict[str, FusionRule] = {
    BASELINE_RULE: _rrf_at(RRF_K),
    "rrf_k20": _rrf_at(20),
    "rrf_k10": _rrf_at(10),
    "rrf_k5": _rrf_at(5),
    "rrf_k1": _rrf_at(1),
    "rrf_k60_sem_x2": _weighted_at(RRF_K, 2.0),
    "rrf_k60_sem_x3": _weighted_at(RRF_K, 3.0),
    "rrf_k10_sem_x2": _weighted_at(10, 2.0),
    "semantic_only": _weighted_at(RRF_K, 1.0, 0.0),
}


def score_rule(rule_name: str, channels: dict[str, QuestionChannels]) -> ablation_eval.ArmResult:
    """Reuses ArmResult so Recall@5, MRR, the per-language split and the
    acceptance gates have exactly one definition, shared with the ablation."""
    rule = RULES[rule_name]
    result = ablation_eval.ArmResult(name=rule_name)
    for qid, channel in channels.items():
        ranked = [cid for cid, _ in rule(channel.semantic_ids, channel.lexical_ids)]
        result.hits[qid] = metrics.recall_at_k(ranked, channel.expected_chunk_ids, _RECALL_K) > 0.0
        result.language_by_id[qid] = channel.language
        result.reciprocal_ranks[qid] = metrics.reciprocal_rank(ranked[:_RECALL_K], channel.expected_chunk_ids)
    return result


def capture_channels(
    questions: Sequence[dict[str, Any]],
    *,
    index_profile: IndexProfile = "contextual-v1",
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, QuestionChannels]:
    retriever = build_retriever("off", expected_profile=index_profile)
    channels: dict[str, QuestionChannels] = {}
    for question in questions:
        text = str(question["question"])
        semantic = retriever._vector_store.query(text, top_n)
        lexical = retriever._lexical_index.query(text, top_n)
        qid = str(question["id"])
        channels[qid] = QuestionChannels(
            qid=qid,
            language=str(question["language"]),
            expected_chunk_ids=list(question["expected_chunk_ids"]),
            semantic_ids=[hit[0] for hit in semantic],
            lexical_ids=[hit[0] for hit in lexical],
        )
    return channels


def _gold_rank_summary(channels: dict[str, QuestionChannels], qids: Sequence[str]) -> list[str]:
    lines = []
    for qid in qids:
        channel = channels.get(qid)
        if channel is None:
            continue
        expected = set(channel.expected_chunk_ids)
        semantic_rank = next((i for i, cid in enumerate(channel.semantic_ids, 1) if cid in expected), None)
        lexical_rank = next((i for i, cid in enumerate(channel.lexical_ids, 1) if cid in expected), None)
        lines.append(f"| `{qid}` | {semantic_rank or '—'} | {lexical_rank or '—'} |")
    return lines


def render_report(
    results: dict[str, ablation_eval.ArmResult],
    version: str,
    index_profile: str,
    channels: Optional[dict[str, QuestionChannels]] = None,
) -> str:
    baseline = results[BASELINE_RULE]
    lines = [
        "# RRF fusion parameter sweep",
        "",
        f"Evaluation set v{version}, live index `{index_profile}`, `expansion_mode=off`, "
        f"Recall@{_RECALL_K} over the answerable subset. Channel outputs captured once per "
        "question and re-fused offline, which is exact — fusion is pure.",
        "",
        "**Nothing here changes a default.** RRF `k=60` and the ascending-`chunk_id` tie-break "
        "are byte-stable invariants; this run reads them and writes nothing. Every row is a "
        "measurement, not a decision.",
        "",
        "| rule | Recall@5 | EN | ES | MRR | new misses | rescues |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, result in results.items():
        delta = ablation_eval.compare(baseline, result)
        lines.append(
            f"| `{name}` | {result.recall_at_5:.3f} | {result.recall_for('en'):.3f} | "
            f"{result.recall_for('es'):.3f} | {result.mrr:.3f} | "
            f"{len(delta.new_misses)} | {len(delta.rescues)} |"
        )

    lines += ["", "## Acceptance gates", ""]
    for name, result in results.items():
        verdicts = [f"{gate}: {'PASS' if check(result) else 'FAIL'}" for gate, check in ablation_eval.GATES]
        delta = ablation_eval.compare(baseline, result)
        zero_new = "PASS" if not delta.new_misses else f"FAIL ({len(delta.new_misses)})"
        lines.append(f"- `{name}` — " + "; ".join(verdicts) + f"; zero new misses: {zero_new}")

    lines += ["", "## Per-question deltas vs the shipped `rrf_k60`", ""]
    for name, result in results.items():
        if name == BASELINE_RULE:
            continue
        delta = ablation_eval.compare(baseline, result)
        lines += [
            f"### `{name}`",
            "",
            f"- Recall@5 delta: {delta.recall_delta:+.4f}",
            f"- New misses (shipped found, rule lost): {delta.new_misses or 'none'}",
            f"- Rescues (shipped lost, rule found): {delta.rescues or 'none'}",
            "",
        ]

    if channels is not None and "semantic_only" in results:
        moved = ablation_eval.compare(baseline, results["semantic_only"]).rescues
        if moved:
            lines += [
                "## Where the gold chunk actually sat",
                "",
                "For every question the shipped fusion loses and the semantic channel alone "
                "retrieves, the gold chunk's rank in each raw channel before fusion:",
                "",
                "| question | semantic rank | BM25 rank |",
                "|---|---|---|",
                *_gold_rank_summary(channels, moved),
                "",
                "A gold chunk sitting at semantic rank 1-3 and landing outside the fused top 5 "
                "is a fusion result, not a retrieval failure.",
                "",
            ]
    return "\n".join(lines) + "\n"


def run(
    rules: Sequence[str] = tuple(RULES),
    *,
    index_profile: IndexProfile = "contextual-v1",
    out_path: Optional[Path] = None,
) -> Path:
    eval_set_integrity.verify()
    data = eval_set_integrity.load_eval_set()
    questions = [q for q in data["questions"] if q["answerable"]]

    channels = capture_channels(questions, index_profile=index_profile)
    results = {name: score_rule(name, channels) for name in rules}
    for name, result in results.items():
        print(
            f"{name}: Recall@5={result.recall_at_5:.3f} "
            f"(en={result.recall_for('en'):.3f}, es={result.recall_for('es'):.3f}) "
            f"MRR={result.mrr:.3f}"
        )

    header = artifacts.resolve_provenance(index_profile, "off")
    report = header.render() + "\n\n" + render_report(results, data["version"], index_profile, channels)

    path = out_path or (EVAL_REPORTS_DIR / f"fusion_sweep_v{data['version']}__{index_profile}__off.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")
    print(f"Report written to: {path}")
    return path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default=",".join(RULES))
    args = parser.parse_args(argv)
    run([r.strip() for r in args.rules.split(",") if r.strip()])


if __name__ == "__main__":
    main()
