from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast

from src.core.config import RefusalPolicyName
from src.domain.models import Citation, DecisionReason, GateBand, RetrievalResult, SourceType

logger = logging.getLogger(__name__)

RRF_K = 60

# Phase 3 grounded-review evidence bounds (ADR-009). A supporting quote must be
# a substantial, contiguous verbatim span — long enough to be real evidence,
# short enough that it can't be the whole chunk pasted back.
MIN_SUPPORTING_QUOTE_CHARS = 40
MAX_SUPPORTING_QUOTE_CHARS = 600

_WHITESPACE_RE = re.compile(r"\s+")

# Domain acronym glossary. Curated 2026-08-30 by scanning corpus/ for all-caps
# tokens by frequency, then looking up each term's expansion.
#   expansions[0] = English expansion — MUST appear verbatim (case-insensitive)
#                   somewhere in corpus/ (test_glossary_english_expansions_are_corpus_attested).
#   expansions[1] = standard Spanish technical rendering — curated, NOT in the
#                   English-only corpus; a plant worker's likely surface form.
GLOSSARY: dict[str, tuple[str, ...]] = {
    "NPSHA": ("net positive suction head available", "altura neta de succión positiva disponible"),
    "NPSHR": ("net positive suction head required", "altura neta de succión positiva requerida"),
    "NPSH": ("net positive suction head", "altura neta de succión positiva"),
    "PEL": ("permissible exposure limit", "límite de exposición permisible"),
    "IDLH": (
        "immediately dangerous to life and health",
        "concentración inmediatamente peligrosa para la vida o la salud",
    ),
    "TWA": ("time-weighted average", "promedio ponderado en el tiempo"),
    "REL": ("recommended exposure limit", "límite de exposición recomendado"),
    "LEL": ("lower explosive limit", "límite inferior de explosividad"),
    "SDS": ("safety data sheet", "hoja de datos de seguridad"),
    "PPE": ("personal protective equipment", "equipo de protección personal"),
    "LOTO": ("lockout/tagout", "bloqueo y etiquetado"),
    "CGMP": ("current good manufacturing practice", "buenas prácticas de manufactura vigentes"),
}

# Longer keys first so the alternation matches "NPSHA" before "NPSH".
_GLOSSARY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(GLOSSARY, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_query(query: str) -> str:
    """Deterministic, corpus-derived acronym expansion applied to the retrieval
    query only — the original query is still what the answer is generated from.
    Returns `query` unchanged when no glossary key is present."""
    matched = {m.group(1).upper() for m in _GLOSSARY_PATTERN.finditer(query)}
    if not matched:
        return query
    lower_query = query.lower()
    additions: list[str] = []
    for key in GLOSSARY:
        if key not in matched:
            continue
        for expansion in GLOSSARY[key]:
            lowered = expansion.lower()
            if lowered in lower_query or any(lowered == a.lower() for a in additions):
                continue
            additions.append(expansion)
    return f"{query} {' '.join(additions)}" if additions else query


def rrf_scores(ranked_chunk_ids: Sequence[str], k: int = RRF_K) -> dict[str, float]:
    return {chunk_id: 1.0 / (k + rank) for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)}


def fuse_rankings(
    semantic_ranked_ids: Sequence[str], bm25_ranked_ids: Sequence[str], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Pure RRF fusion over two already-ranked id lists. Returns (chunk_id,
    fused_score) pairs sorted by (-fused_score, chunk_id) — the tie-break is
    part of the policy, not the orchestration. No metadata, no I/O."""
    semantic_rrf = rrf_scores(semantic_ranked_ids, k)
    bm25_rrf = rrf_scores(bm25_ranked_ids, k)
    all_ids = set(semantic_ranked_ids) | set(bm25_ranked_ids)
    fused = [(cid, semantic_rrf.get(cid, 0.0) + bm25_rrf.get(cid, 0.0)) for cid in all_ids]
    fused.sort(key=lambda pair: (-pair[1], pair[0]))
    return fused


def is_confident(top1_semantic_score: Optional[float], threshold: float) -> bool:
    return top1_semantic_score is not None and top1_semantic_score >= threshold


def top1_semantic_score_from_results(results: Sequence[RetrievalResult]) -> Optional[float]:
    """Returns None instead of raising on an empty/scoreless list, because
    here that's a legitimate "nothing relevant retrieved" signal that should
    flow into a refusal, not crash the /query endpoint."""
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


class RefusalPolicy:
    def __init__(
        self,
        threshold: float,
        *,
        mode: RefusalPolicyName = "binary",
        review_floor: float = 0.5500,
    ) -> None:
        for name, value in (("threshold", threshold), ("review_floor", review_floor)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"RefusalPolicy {name} must be a finite value in [0, 1], got {value!r}")
        if mode not in ("binary", "grounded_review"):
            raise ValueError(f"RefusalPolicy mode must be 'binary' or 'grounded_review', got {mode!r}")
        if mode == "grounded_review" and not review_floor < threshold:
            raise ValueError(
                f"RefusalPolicy review_floor must be strictly below threshold for grounded_review "
                f"(floor={review_floor}, threshold={threshold})"
            )
        self._threshold = threshold
        self._mode: RefusalPolicyName = mode
        self._review_floor = review_floor

    def top1_semantic_score(self, results: Sequence[RetrievalResult]) -> Optional[float]:
        return top1_semantic_score_from_results(results)

    def classify_score(self, score: Optional[float]) -> GateBand:
        if score is None:
            return "hard_refuse"
        if self._mode == "binary":
            return "confident" if score >= self._threshold else "hard_refuse"
        if score < self._review_floor:
            return "hard_refuse"
        if score < self._threshold:
            return "grounded_review"
        return "confident"

    def classify(self, results: Sequence[RetrievalResult]) -> GateBand:
        return self.classify_score(self.top1_semantic_score(results))

    def hard_refuse_reason(self) -> DecisionReason:
        return "below_review_floor" if self._mode == "grounded_review" else "below_binary_threshold"

    def is_confident(self, results: Sequence[RetrievalResult]) -> bool:
        return self.classify(results) == "confident"


def normalize_evidence_text(value: str) -> str:
    """NFKC + whitespace-collapse. The only tolerance allowed on a verbatim
    quote (ADR-009): line breaks and repeated spaces in the source markdown
    must not defeat an otherwise-exact match. No case folding, no punctuation
    changes, no elision."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


@dataclass(frozen=True)
class GroundingValidation:
    citations: list[Citation]
    failure_reason: Optional[DecisionReason]


class GroundedEvidenceResolver:
    """Fail-closed: any invalid evidence item invalidates the whole response
    (no partial acceptance). Returns resolved citations only when every item
    names a retrieved chunk and carries a normalized 40-600 char verbatim
    substring of that chunk's raw text."""

    @staticmethod
    def resolve(
        evidence_items: Sequence[Any], results: Sequence[RetrievalResult]
    ) -> GroundingValidation:
        if not evidence_items:
            return GroundingValidation([], "missing_evidence")

        results_by_id = {result.chunk_id: result for result in results}
        cited_ids: list[str] = []
        seen: set[str] = set()

        for item in evidence_items:
            if not isinstance(item, dict):
                return GroundingValidation([], "invalid_evidence_shape")
            chunk_id = item.get("chunk_id")
            quote = item.get("supporting_quote")
            if not isinstance(chunk_id, str) or not isinstance(quote, str):
                return GroundingValidation([], "invalid_evidence_shape")

            result = results_by_id.get(chunk_id)
            if result is None:
                return GroundingValidation([], "chunk_not_retrieved")

            normalized_quote = normalize_evidence_text(quote)
            if len(normalized_quote) < MIN_SUPPORTING_QUOTE_CHARS:
                return GroundingValidation([], "quote_too_short")
            if len(normalized_quote) > MAX_SUPPORTING_QUOTE_CHARS:
                return GroundingValidation([], "quote_too_long")

            chunk_text = result.metadata.get("chunk_text")
            if not isinstance(chunk_text, str):
                return GroundingValidation([], "quote_not_found")
            if normalized_quote not in normalize_evidence_text(chunk_text):
                return GroundingValidation([], "quote_not_found")

            if chunk_id not in seen:
                seen.add(chunk_id)
                cited_ids.append(chunk_id)

        resolution = CitationResolver.resolve([{"chunk_id": cid} for cid in cited_ids], results)
        if resolution.failure_reason is not None or not resolution.citations:
            return GroundingValidation([], "unresolved_citation")
        return GroundingValidation(resolution.citations, None)


@dataclass(frozen=True)
class CitationResolution:
    citations: list[Citation]
    failure_reason: Optional[DecisionReason]


# Every field a Citation carries besides chunk_id is looked up here, from the
# real retrieved chunk. None of them is ever read from LLM output.
_CITATION_METADATA_FIELDS = (
    "document_id",
    "document_title",
    "section_heading",
    "revision",
    "source_type",
)


class CitationResolver:
    """Fail-closed, like GroundedEvidenceResolver: one citation naming a chunk
    that wasn't retrieved — or a retrieved chunk whose metadata is missing a
    citation field — invalidates the whole set. Partial resolution used to
    drop the bad entry and serve the rest, which quietly presented a
    partly-unverifiable answer as fully cited."""

    @staticmethod
    def resolve(
        llm_citations: Sequence[Any], results: Sequence[RetrievalResult]
    ) -> CitationResolution:
        results_by_chunk_id = {result.chunk_id: result for result in results}
        resolved: list[Citation] = []

        for llm_citation in llm_citations:
            if not isinstance(llm_citation, dict):
                return CitationResolver._reject("invalid_citation_shape", None)
            chunk_id = llm_citation.get("chunk_id")
            if not isinstance(chunk_id, str):
                return CitationResolver._reject("invalid_citation_shape", None)

            result = results_by_chunk_id.get(chunk_id)
            if result is None:
                return CitationResolver._reject("citation_not_in_retrieved_set", chunk_id)

            metadata = result.metadata
            fields: dict[str, str] = {}
            for field_name in _CITATION_METADATA_FIELDS:
                value = metadata.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    return CitationResolver._reject("citation_metadata_incomplete", chunk_id)
                fields[field_name] = value
            if fields["source_type"] not in ("public", "synthetic"):
                return CitationResolver._reject("citation_metadata_incomplete", chunk_id)

            resolved.append(
                Citation(
                    document_id=fields["document_id"],
                    document_title=fields["document_title"],
                    section_heading=fields["section_heading"],
                    revision=fields["revision"],
                    chunk_id=result.chunk_id,
                    source_type=cast(SourceType, fields["source_type"]),
                )
            )
        return CitationResolution(resolved, None)

    @staticmethod
    def _reject(event: str, chunk_id: Optional[str]) -> CitationResolution:
        """Logs the event and the offending chunk_id only — never the citation
        payload, the answer, or any chunk body."""
        logger.warning(
            "citation could not be resolved from retrieved metadata; rejecting the whole set",
            extra={"event": event, "chunk_id": chunk_id},
        )
        return CitationResolution([], "unresolved_citation")
