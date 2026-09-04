"""Retrieval-channel ablation (audit bucket 5).

Answers one question with evidence instead of belief: does the BM25 channel
contribute anything? The audit recorded "BM25 aporta ~0" as an unproven nuance
— it produced no exclusive top-5 rescues in ES, which is not an ablation.

Default-off by construction: this module never writes an index, never changes
a served default, and never calls an LLM. It reads the live index and the
frozen evaluation set, and writes a report.

Acceptance gates for any candidate replacing contextual-v1/off (all four, and
even then the flip is a separate owner decision): EN Recall@5 >= 0.917,
ES Recall@5 >= 0.844, global Recall@5 >= 0.887, and ZERO new misses. Aggregate
recall alone cannot satisfy the last one, so `compare` names the questions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from src.adapters.secondary.lexical.null_lexical_index import NullLexicalIndex
from src.core.paths import EVAL_REPORTS_DIR, RETRIEVAL_OUTPUT_DIR
from src.domain.models import IndexProfile
from src.features.evaluation import artifacts, eval_set_integrity, metrics
from src.features.evaluation._eval_retriever import build_retriever
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K, HybridRetriever

BASELINE_ARM = "hybrid_word_lower"
ARMS: tuple[str, ...] = (BASELINE_ARM, "semantic_only", "hybrid_snowball_bilingual")

# Experiment indexes live here, never at settings.bm25_path. Gitignored.
EXPERIMENT_DIR = RETRIEVAL_OUTPUT_DIR / "experiments"

_RECALL_K = 5


@dataclass
class ArmResult:
    name: str
    hits: dict[str, bool] = field(default_factory=dict)
    language_by_id: dict[str, str] = field(default_factory=dict)
    reciprocal_ranks: dict[str, float] = field(default_factory=dict)

    @property
    def recall_at_5(self) -> float:
        return (sum(self.hits.values()) / len(self.hits)) if self.hits else 0.0

    @property
    def mrr(self) -> float:
        return metrics.mean_reciprocal_rank(list(self.reciprocal_ranks.values()))

    def recall_for(self, language: str) -> float:
        subset = [hit for qid, hit in self.hits.items() if self.language_by_id.get(qid) == language]
        return (sum(subset) / len(subset)) if subset else 0.0


@dataclass(frozen=True)
class ArmDelta:
    baseline: str
    candidate: str
    new_misses: list[str]
    rescues: list[str]
    recall_delta: float


def compare(baseline: ArmResult, candidate: ArmResult) -> ArmDelta:
    """Equal recall can hide a swap: one question won, another lost. The
    zero-new-misses gate is about the identity of the questions, not the
    count, so both lists are named."""
    shared = sorted(set(baseline.hits) & set(candidate.hits))
    return ArmDelta(
        baseline=baseline.name,
        candidate=candidate.name,
        new_misses=[q for q in shared if baseline.hits[q] and not candidate.hits[q]],
        rescues=[q for q in shared if not baseline.hits[q] and candidate.hits[q]],
        recall_delta=candidate.recall_at_5 - baseline.recall_at_5,
    )


def _snowball_index() -> Any:
    try:
        from src.adapters.secondary.lexical.snowball_bm25_index import SnowballBm25Index
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the hybrid_snowball_bilingual arm needs the experiment-only dependency: "
            "pip install -r requirements-experiments.txt"
        ) from exc
    path = EXPERIMENT_DIR / "snowball_bm25.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} not found — build it first with "
            "`python -m src.features.evaluation.ablation_eval --build-snowball`"
        )
    return SnowballBm25Index(persist_path=path)


def build_arm(
    arm: str,
    *,
    index_profile: IndexProfile = "contextual-v1",
    verify_physical_coherence: bool = True,
) -> HybridRetriever:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}; expected one of {ARMS}")
    base = build_retriever(
        "off", expected_profile=index_profile, verify_physical_coherence=verify_physical_coherence
    )
    if arm == BASELINE_ARM:
        return base
    lexical = NullLexicalIndex() if arm == "semantic_only" else _snowball_index()
    # Reaching into the built retriever's vector store is deliberate and scoped
    # to offline evaluation: the arms must share ONE vector channel so the only
    # thing that varies is the lexical one. HybridRetriever gets no public
    # accessor for this — serving has no reason to swap a channel.
    return HybridRetriever(base._vector_store, lexical, expansion_mode="off")


def score_arm(arm: str, retriever: HybridRetriever, questions: Sequence[dict[str, Any]]) -> ArmResult:
    result = ArmResult(name=arm)
    for question in questions:
        results = retriever.retrieve(question["question"], k=SEMANTIC_EXTRACTION_K)
        retrieved = [r.chunk_id for r in results]
        expected = question["expected_chunk_ids"]
        qid = question["id"]
        result.hits[qid] = metrics.recall_at_k(retrieved, expected, _RECALL_K) > 0.0
        result.language_by_id[qid] = question["language"]
        result.reciprocal_ranks[qid] = metrics.reciprocal_rank(retrieved[:_RECALL_K], expected)
    return result


# The audit's acceptance criteria, verbatim. A candidate that misses any of
# these does not replace contextual-v1/off, and clearing them all is still not
# sufficient — zero new misses and a separate owner decision are also required.
# The thresholds are the PUBLISHED three-decimal figures, and the baseline's own
# values round up into them: EN is 44/48 = 0.91666..., ES is 27/32 = 0.84375.
# Comparing raw floats would therefore fail the shipped baseline against its own
# gate, so the comparison rounds the same way the thresholds were stated.
def _at_least(value: float, threshold: float) -> bool:
    return round(value, 3) >= threshold


GATES: tuple[tuple[str, Callable[[ArmResult], bool]], ...] = (
    ("EN Recall@5 >= 0.917", lambda r: _at_least(r.recall_for("en"), 0.917)),
    ("ES Recall@5 >= 0.844", lambda r: _at_least(r.recall_for("es"), 0.844)),
    ("Global Recall@5 >= 0.887", lambda r: _at_least(r.recall_at_5, 0.887)),
)


def render_report(results: dict[str, ArmResult], version: str, index_profile: str) -> str:
    baseline = results[BASELINE_ARM]
    lines = [
        "# Retrieval channel ablation",
        "",
        f"Evaluation set v{version}, live index `{index_profile}`, `expansion_mode=off`, "
        f"Recall@{_RECALL_K} over the answerable subset.",
        "",
        "**Nothing here changes a default.** A candidate replaces contextual-v1/off only if it "
        "clears every gate below AND introduces zero new misses AND the owner separately approves.",
        "",
        "| arm | Recall@5 | EN | ES | MRR | new misses | rescues |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, result in results.items():
        delta = compare(baseline, result)
        lines.append(
            f"| `{arm}` | {result.recall_at_5:.3f} | {result.recall_for('en'):.3f} | "
            f"{result.recall_for('es'):.3f} | {result.mrr:.3f} | "
            f"{len(delta.new_misses)} | {len(delta.rescues)} |"
        )

    lines += ["", "## Acceptance gates", ""]
    for arm, result in results.items():
        verdicts = [f"{name}: {'PASS' if check(result) else 'FAIL'}" for name, check in GATES]
        delta = compare(baseline, result)
        zero_new = "PASS" if not delta.new_misses else f"FAIL ({len(delta.new_misses)})"
        lines.append(f"- `{arm}` — " + "; ".join(verdicts) + f"; zero new misses: {zero_new}")

    lines += ["", "## Per-question deltas vs the current hybrid baseline", ""]
    for arm, result in results.items():
        if arm == BASELINE_ARM:
            continue
        delta = compare(baseline, result)
        lines += [
            f"### `{arm}`",
            "",
            f"- Recall@5 delta: {delta.recall_delta:+.4f}",
            f"- New misses (baseline found, candidate lost): {delta.new_misses or 'none'}",
            f"- Rescues (baseline lost, candidate found): {delta.rescues or 'none'}",
            "",
        ]

    if BASELINE_ARM in results and "semantic_only" in results:
        semantic = results["semantic_only"]
        exclusive = compare(semantic, results[BASELINE_ARM]).rescues
        cost = compare(baseline, semantic).rescues
        lines += [
            "## Does BM25 contribute anything?",
            "",
            f"- Questions the hybrid arm gets that semantic-only misses: {exclusive or 'none'}.",
            f"- Questions semantic-only gets that the hybrid arm misses: {cost or 'none'}.",
            "",
        ]
        if exclusive:
            lines += [
                "The lexical channel earns exclusive top-5 rescues on this set, so "
                "'BM25 aporta ~0' is refuted in its favour.",
                "",
            ]
        elif cost:
            lines += [
                "The lexical channel earns **no** exclusive rescue, and RRF fusion demotes "
                f"{len(cost)} question(s) the semantic channel alone retrieves. On this "
                "evaluation set BM25 is therefore not neutral, as the audit's unproven "
                "'BM25 aporta ~0' nuance supposed — it is net-negative.",
                "",
                "This is a measurement, not a decision. Removing or down-weighting the lexical "
                "channel is a separate owner call, and one evaluation set of 80 answerable "
                "questions over a 14-document corpus is thin evidence for a permanent "
                "architectural change. The obvious confounder to rule out first is RRF's "
                "rank-only fusion (k=60): it weights a BM25 rank-1 hit identically however weak "
                "its score is, so a near-miss lexical match can outrank a strong semantic one.",
                "",
            ]
        else:
            lines += [
                "The lexical channel earns no exclusive rescue and costs nothing on this set — "
                "consistent with the audit's 'BM25 aporta ~0' nuance, and still not a licence "
                "to remove it without a separate decision.",
                "",
            ]
    return "\n".join(lines) + "\n"


def run(
    arms: Sequence[str] = ARMS,
    *,
    index_profile: IndexProfile = "contextual-v1",
    out_path: Optional[Path] = None,
) -> Path:
    eval_set_integrity.verify()
    data = eval_set_integrity.load_eval_set()
    questions = [q for q in data["questions"] if q["answerable"]]

    results: dict[str, ArmResult] = {}
    for arm in arms:
        retriever = build_arm(arm, index_profile=index_profile)
        results[arm] = score_arm(arm, retriever, questions)
        print(
            f"{arm}: Recall@5={results[arm].recall_at_5:.3f} "
            f"(en={results[arm].recall_for('en'):.3f}, es={results[arm].recall_for('es'):.3f}) "
            f"MRR={results[arm].mrr:.3f}"
        )

    header = artifacts.resolve_provenance(index_profile, "off")
    report = header.render() + "\n\n" + render_report(results, data["version"], index_profile)

    path = out_path or (EVAL_REPORTS_DIR / f"ablation_v{data['version']}__{index_profile}__off.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")
    print(f"Report written to: {path}")
    return path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=",".join((BASELINE_ARM, "semantic_only")))
    parser.add_argument("--build-snowball", action="store_true")
    args = parser.parse_args(argv)

    if args.build_snowball:
        _build_snowball_index()
        return
    run([a.strip() for a in args.arms.split(",") if a.strip()])


def _build_snowball_index() -> Path:
    """Builds the experiment-only Snowball index under retrieval/output/experiments/.
    Never touches settings.bm25_path — asserted, not just intended."""
    from src.adapters.secondary.lexical.snowball_bm25_index import SnowballBm25Index
    from src.core.config import load_settings
    from src.features.retrieval import index_manifest
    from src.features.retrieval.chunk_store import load_chunks

    path = EXPERIMENT_DIR / "snowball_bm25.json"
    assert path != load_settings().bm25_path, "experiment index must not overwrite the live one"
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    SnowballBm25Index(persist_path=path).build_index(
        chunks, chunks_sha256=index_manifest.chunks_sha256()
    )
    print(f"Snowball experiment index written to: {path} ({len(chunks)} chunks)")
    return path


if __name__ == "__main__":
    main()
