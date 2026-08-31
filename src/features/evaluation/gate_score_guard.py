from __future__ import annotations

import sys
from typing import Optional

from src.domain.models import ExpansionMode, IndexProfile
from src.domain.policies import top1_semantic_score_from_results
from src.domain.ports import RetrieverPort
from src.features.evaluation import regression_set_integrity
from src.features.evaluation._eval_retriever import assert_live_index_profile, build_retriever
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

# The Phase 3 grounded-review band. Pre-registered before the holdout was
# authored: min(r001, r002) top1_semantic on contextual-v1/off is ~0.564, so a
# floor of 0.5500 keeps both known false-refusals in the band with margin
# >= 0.0142 while leaving r019/r020 hard-refused. The guard checks the BAND,
# never an exact cosine — those can drift a little between Windows/Linux.
GATE_REVIEW_FLOOR = 0.5500
GATE_CONFIDENT_THRESHOLD = 0.5999

EXPECTED_GROUNDED_REVIEW_IDS: tuple[str, ...] = ("r001", "r002")


def score_in_review_band(score: Optional[float]) -> bool:
    return score is not None and GATE_REVIEW_FLOOR <= score < GATE_CONFIDENT_THRESHOLD


def run(
    *,
    index_profile: IndexProfile = "contextual-v1",
    expansion_mode: ExpansionMode = "off",
    retriever: Optional[RetrieverPort] = None,
) -> None:
    """CI guard: the two reported bilingual false-refusals (r001/r002) must land
    in the grounded-review band on the freshly-built index, or the Phase 3
    floor no longer matches reality and the plan must stop and re-measure."""
    regression_set_integrity.verify()
    queries_by_id = {q["id"]: q for q in regression_set_integrity.load_regression_set()["queries"]}

    if retriever is None:
        assert_live_index_profile(index_profile)
        retriever = build_retriever(expansion_mode)

    failures: list[str] = []
    for qid in EXPECTED_GROUNDED_REVIEW_IDS:
        query = queries_by_id[qid]
        results = retriever.retrieve(query["query"], k=SEMANTIC_EXTRACTION_K)
        score = top1_semantic_score_from_results(results)
        band_ok = score_in_review_band(score)
        rendered = "n/a" if score is None else f"{score:.4f}"
        print(f"{qid} ({query['language']}): top1_semantic={rendered} in_review_band={band_ok}")
        if not band_ok:
            failures.append(
                f"{qid} top1_semantic {rendered} outside "
                f"[{GATE_REVIEW_FLOOR}, {GATE_CONFIDENT_THRESHOLD})"
            )

    if failures:
        print("GATE SCORE GUARD FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "  Re-measure contextual-v1/off on this platform; do NOT retune the "
            "floor to chase individual cases.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("gate score guard OK - r001/r002 both in the grounded-review band")


if __name__ == "__main__":
    run()
