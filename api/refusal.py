from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_confident(top1_semantic_score: Optional[float], threshold: float) -> bool:
    return top1_semantic_score is not None and top1_semantic_score >= threshold


def top1_semantic_score_from_results(results: list) -> Optional[float]:
    """Live-path counterpart to eval.metrics.top1_semantic_score: returns None
    instead of raising on an empty/scoreless list, because here that's a
    legitimate "nothing relevant retrieved" signal that should flow into a
    refusal, not crash the /query endpoint."""
    for result in results:
        if result.semantic_rank == 1:
            return result.semantic_score
    scored = [result.semantic_score for result in results if result.semantic_score is not None]
    if not scored:
        return None
    fallback = max(scored)
    logger.warning(
        "no result with semantic_rank == 1 among %d results; falling back to max semantic_score (%.4f)",
        len(results),
        fallback,
    )
    return fallback


def is_confident_for_results(results: list, threshold: float) -> bool:
    return is_confident(top1_semantic_score_from_results(results), threshold)
