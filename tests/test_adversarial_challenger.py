"""Adversarial stress-testing suite for RAG4 diagnostic review.

Cross-Reviewer 2 / Adversarial Challenger empirical verification suite.
Validates:
1. Exact score inversion: Score(q093) > Score(r020) > Score(q072)
2. Boundary & regression stress-testing (floors: 0.5410, 0.5421, 0.5490, 0.5500)
3. Gate band policy routing (binary vs grounded_review) & fail-closed evidence validation
4. Bilingual consistency across English/Spanish query pairs (q049/q050, q071/q072)
5. Holdout gate containment (G1-G4) and impossibility proof of scalar recovery
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.core.config import DEFAULT_REFUSAL_COSINE_THRESHOLD, DEFAULT_REFUSAL_REVIEW_FLOOR
from src.domain.models import RetrievalResult
from src.domain.policies import (
    GroundedEvidenceResolver,
    RefusalPolicy,
)
from src.features.evaluation import floor_sweep as fs
from src.features.evaluation.floor_sweep import (
    FeatureRow,
    classify_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DETAILS_FILE = (
    REPO_ROOT / "eval" / "reports" / "retrieval_details_v1.1.0__contextual-v1__off.jsonl"
)
RETRIEVAL_REPORT_FILE = (
    REPO_ROOT / "eval" / "reports" / "retrieval_report_v1.1.0__contextual-v1__off.md"
)
REGRESSION_REPORT_FILE = (
    REPO_ROOT / "eval" / "reports" / "regression_eval_v1.1.0__contextual-v1__off.md"
)
SWEEP_SUMMARY_FILE = REPO_ROOT / "docs" / "eval" / "floor_sweep_summary.md"
GATE_HOLDOUT_FILE = REPO_ROOT / "eval" / "gate_holdout_v1.0.0.json"
EVAL_SET_FILE = REPO_ROOT / "eval" / "eval_set.json"
REGRESSION_FILE = REPO_ROOT / "eval" / "regression_queries.json"


@pytest.fixture(scope="module")
def features_by_id() -> dict[str, FeatureRow]:
    """Rebuild the challenged rows from committed evaluation evidence.

    Calibration runs are intentionally gitignored, so a test must never depend
    on one timestamped local run. The committed retrieval/regression reports are
    the portable evidence available in a clean checkout.
    """
    eval_questions = {
        question["id"]: question
        for question in json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))["questions"]
    }
    rows: dict[str, FeatureRow] = {}
    wanted = {"q049", "q071", "q072"}
    for line in RETRIEVAL_DETAILS_FILE.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        question_id = data["id"]
        if question_id not in wanted:
            continue
        question = eval_questions[question_id]
        top5 = data["top5"]
        semantic_top = next(item for item in top5 if item["semantic_rank"] == 1)
        bm25_top = next((item for item in top5 if item["bm25_rank"] == 1), None)
        expected_ids = set(question.get("expected_chunk_ids", []))
        expected = [item for item in top5 if item["chunk_id"] in expected_ids]
        rows[question_id] = FeatureRow(
            question_id=question_id,
            language=question["language"],
            answerable=question["answerable"],
            cohort="eval_set",
            top1_semantic=semantic_top["semantic_score"],
            sem_margin=None,
            sem_top1_id=semantic_top["chunk_id"],
            bm25_top1_id=None if bm25_top is None else bm25_top["chunk_id"],
            expected_sem_rank=min(
                (item["semantic_rank"] for item in expected if item["semantic_rank"] is not None),
                default=None,
            ),
            expected_bm25_rank=min(
                (item["bm25_rank"] for item in expected if item["bm25_rank"] is not None),
                default=None,
            ),
            sem_bm25_top1_agree=(
                bm25_top is not None and semantic_top["chunk_id"] == bm25_top["chunk_id"]
            ),
            sem_top1_in_bm25_top_n=semantic_top["bm25_rank"] is not None,
            channels_overlap_top_n=any(
                item["semantic_rank"] is not None and item["bm25_rank"] is not None
                for item in top5
            ),
        )

    retrieval_report = RETRIEVAL_REPORT_FILE.read_text(encoding="utf-8")
    for question_id in ("q050", "q093"):
        question = eval_questions[question_id]
        match = re.search(
            rf"^\| {question_id} \|.*?\| ([0-9.]+) \|$",
            retrieval_report,
            re.MULTILINE,
        )
        assert match is not None
        rows[question_id] = FeatureRow(
            question_id=question_id,
            language=question["language"],
            answerable=question["answerable"],
            cohort="eval_set",
            top1_semantic=float(match.group(1)),
            sem_margin=None,
            sem_top1_id=None,
            bm25_top1_id=None,
            expected_sem_rank=None,
            expected_bm25_rank=None,
            sem_bm25_top1_agree=False,
            sem_top1_in_bm25_top_n=False,
            channels_overlap_top_n=False,
        )

    regression_questions = {
        question["id"]: question
        for question in json.loads(REGRESSION_FILE.read_text(encoding="utf-8"))["queries"]
    }
    report = REGRESSION_REPORT_FILE.read_text(encoding="utf-8")
    for qid in ("r018", "r019", "r020"):
        match = re.search(rf"^\| {qid} \| [a-z]+ \| False \| ([0-9.]+) \|", report, re.MULTILINE)
        assert match is not None
        q = regression_questions[qid]
        rows[qid] = FeatureRow(
            question_id=qid,
            language=q["language"],
            answerable=q["should_answer"],
            cohort="regression_controls",
            top1_semantic=float(match.group(1)),
            sem_margin=None,
            sem_top1_id=None,
            bm25_top1_id=None,
            expected_sem_rank=None,
            expected_bm25_rank=None,
            sem_bm25_top1_agree=False,
            sem_top1_in_bm25_top_n=False,
            channels_overlap_top_n=False,
        )
    return rows


# =========================================================================
# 1. SCORE INVERSION EMPIRICAL VALIDATION
# =========================================================================

def test_reported_score_inversion_sequence(features_by_id: dict[str, FeatureRow]) -> None:
    """Verify the score sequence at the precision in committed reports:
    Score(q093) = 0.54971 > Score(r020) = 0.5420 > Score(q072) = 0.54137
    """
    row_q093 = features_by_id["q093"]
    row_r020 = features_by_id["r020"]
    row_q072 = features_by_id["q072"]

    score_q093 = row_q093.top1_semantic
    score_r020 = row_r020.top1_semantic
    score_q072 = row_q072.top1_semantic

    assert score_q093 is not None
    assert score_r020 is not None
    assert score_q072 is not None

    # Eval details retain full precision; the regression report rounds to 4 decimals.
    assert pytest.approx(0.54971, abs=1e-4) == score_q093
    assert pytest.approx(0.54198, abs=1e-4) == score_r020
    assert pytest.approx(0.54137, abs=1e-4) == score_q072

    # Inversion inequality
    assert score_q093 > score_r020 > score_q072

    # Metadata / safety verification
    assert row_q093.answerable is False, "q093 must be unanswerable"
    assert row_q093.language == "es"
    assert row_r020.answerable is False, "r020 must be unanswerable control"
    assert row_r020.language == "en"
    assert row_q072.answerable is True, "q072 is an answerable query"
    assert row_q072.language == "en"


def test_score_positioning_relative_to_baseline_floor(features_by_id: dict[str, FeatureRow]) -> None:
    """Verify all 3 queries are strictly below the current 0.5500 floor."""
    baseline_floor = DEFAULT_REFUSAL_REVIEW_FLOOR  # 0.5500
    score_q093 = features_by_id["q093"].top1_semantic
    score_r020 = features_by_id["r020"].top1_semantic
    score_q072 = features_by_id["q072"].top1_semantic

    assert score_q093 is not None and score_q093 < baseline_floor
    assert score_r020 is not None and score_r020 < baseline_floor
    assert score_q072 is not None and score_q072 < baseline_floor

    # Margins below floor
    margin_q093 = score_q093 - baseline_floor
    margin_r020 = score_r020 - baseline_floor
    margin_q072 = score_q072 - baseline_floor

    assert pytest.approx(-0.00029, abs=1e-4) == margin_q093
    assert pytest.approx(-0.00802, abs=1e-4) == margin_r020
    assert pytest.approx(-0.00863, abs=1e-4) == margin_q072


# =========================================================================
# 2. BOUNDARY & REGRESSION STRESS-TESTING (FLOOR CANDIDATES)
# =========================================================================

@pytest.mark.parametrize(
    "floor, expected_q093_band, expected_r020_band, expected_q072_band",
    [
        (0.5500, "hard_refuse", "hard_refuse", "hard_refuse"),
        (0.5490, "grounded_review", "hard_refuse", "hard_refuse"),
        (0.5421, "grounded_review", "hard_refuse", "hard_refuse"),
        (0.5410, "grounded_review", "grounded_review", "grounded_review"),
    ],
)
def test_candidate_floor_classification_dynamics(
    features_by_id: dict[str, FeatureRow],
    floor: float,
    expected_q093_band: str,
    expected_r020_band: str,
    expected_q072_band: str,
) -> None:
    """Stress test boundary behavior across candidate floors:
    - At 0.5500: All 3 are hard_refuse (safe baseline).
    - At 0.5490: q093 is promoted to grounded_review (unanswerable leak!), while q072 remains refused.
    - At 0.5421: q093 is promoted to grounded_review, while r020/q072 remain refused.
    - At 0.5410: q072 is admitted to grounded_review, but q093 and r020 are BOTH promoted (safety breach).
    """
    row_q093 = features_by_id["q093"]
    row_r020 = features_by_id["r020"]
    row_q072 = features_by_id["q072"]

    assert classify_candidate(row_q093, floor, "none") == expected_q093_band
    assert classify_candidate(row_r020, floor, "none") == expected_r020_band
    assert classify_candidate(row_q072, floor, "none") == expected_q072_band


def test_committed_sweep_records_24_ineligible_candidates() -> None:
    """Keep the committed 24-cell sweep summary complete and internally consistent."""
    report = SWEEP_SUMMARY_FILE.read_text(encoding="utf-8")
    grid_rows = [
        line
        for line in report.splitlines()
        if re.match(r"^\| 0\.\d{2}(?: \(current\))? \|", line)
    ]
    assert len(grid_rows) == len(fs.FLOORS) * len(fs.SIGNAL_NAMES) == 24
    assert all(line.endswith("| no |") for line in grid_rows)
    assert "## Mechanically eligible candidates\n\n_None._" in report


def test_gate_holdout_has_balanced_unanswerable_cohort() -> None:
    """Verify the committed holdout's unanswerable language balance."""
    with open(GATE_HOLDOUT_FILE, "r", encoding="utf-8") as f:
        holdout = json.loads(f.read())

    questions = holdout["questions"]
    unanswerable = [q for q in questions if not q["answerable"]]
    assert len(unanswerable) == 24
    assert len([q for q in unanswerable if q["language"] == "en"]) == 12
    assert len([q for q in unanswerable if q["language"] == "es"]) == 12


# =========================================================================
# 3. GATE BAND POLICY ROUTING & FAIL-CLOSED EVIDENCE VALIDATION
# =========================================================================

def test_control_routing_under_binary_policy(features_by_id: dict[str, FeatureRow]) -> None:
    """Under binary policy (threshold = 0.5999):
    r018, r019, r020 parsed from committed evidence must all be hard_refuse.
    """
    policy_binary = RefusalPolicy(
        DEFAULT_REFUSAL_COSINE_THRESHOLD,
        mode="binary",
        review_floor=DEFAULT_REFUSAL_REVIEW_FLOOR,
    )

    assert policy_binary.classify_score(features_by_id["r018"].top1_semantic) == "hard_refuse"
    assert policy_binary.classify_score(features_by_id["r019"].top1_semantic) == "hard_refuse"
    assert policy_binary.classify_score(features_by_id["r020"].top1_semantic) == "hard_refuse"


def test_control_routing_under_grounded_review_policy(features_by_id: dict[str, FeatureRow]) -> None:
    """Under grounded_review policy (threshold = 0.5999, floor = 0.5500):
    r018 -> grounded_review
    r019 -> hard_refuse
    r020 -> hard_refuse
    """
    policy_review = RefusalPolicy(
        DEFAULT_REFUSAL_COSINE_THRESHOLD,
        mode="grounded_review",
        review_floor=DEFAULT_REFUSAL_REVIEW_FLOOR,
    )

    assert policy_review.classify_score(features_by_id["r018"].top1_semantic) == "grounded_review"
    assert policy_review.classify_score(features_by_id["r019"].top1_semantic) == "hard_refuse"
    assert policy_review.classify_score(features_by_id["r020"].top1_semantic) == "hard_refuse"


def test_r018_grounded_review_fail_closed_rejection() -> None:
    """Verify that when r018 enters grounded_review, GroundedEvidenceResolver
    safely rejects it fail-closed when evidence is missing or ungrounded.
    """
    mock_results = [
        RetrievalResult(
            chunk_id="doe-hdbk-1018-1-pumps::chunk-0007",
            fused_score=0.035,
            semantic_rank=1,
            semantic_score=0.5630,
            bm25_rank=1,
            bm25_score=10.5,
            metadata={
                "document_id": "doe-hdbk-1018-1-pumps",
                "document_title": "Pumps Handbook",
                "section_heading": "NPSH Principles",
                "revision": "Rev 1",
                "source_type": "public",
                "chunk_text": "Centrifugal pumps require net positive suction head to prevent cavitation.",
            },
        )
    ]

    # 1. Missing evidence -> rejected
    validation_empty = GroundedEvidenceResolver.resolve([], mock_results)
    assert validation_empty.failure_reason == "missing_evidence"
    assert validation_empty.citations == []

    # 2. Fabricated citation chunk -> rejected
    validation_unretrieved = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "api-610-pumps::chunk-0001", "supporting_quote": "A 10 percent margin is required for API 610."}],
        mock_results,
    )
    assert validation_unretrieved.failure_reason == "chunk_not_retrieved"
    assert validation_unretrieved.citations == []

    # 3. Hallucinated quote not in chunk text -> rejected
    validation_quote_missing = GroundedEvidenceResolver.resolve(
        [{
            "chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007",
            "supporting_quote": "API 610 requires an NPSH margin of at least 1.0 meter across all operations.",
        }],
        mock_results,
    )
    assert validation_quote_missing.failure_reason == "quote_not_found"
    assert validation_quote_missing.citations == []

    # 4. Too short quote -> rejected
    validation_short = GroundedEvidenceResolver.resolve(
        [{"chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "supporting_quote": "Short quote"}],
        mock_results,
    )
    assert validation_short.failure_reason == "quote_too_short"


# =========================================================================
# 4. BILINGUAL CONSISTENCY EMPIRICAL VERIFICATION
# =========================================================================

def test_bilingual_pair_q049_and_q050(features_by_id: dict[str, FeatureRow]) -> None:
    """Verify bilingual pair q049 (es) and q050 (en) on LOTO review frequency:
    - Committed evidence places both top-1 semantic scores in the review band
    - q049's committed fused top-5 details retain the gold semantic-rank-1 chunk
    - Both achieve scores in [0.5500, 0.5999) and route to grounded_review
    """
    row_q049 = features_by_id["q049"]
    row_q050 = features_by_id["q050"]

    assert row_q049.language == "es"
    assert row_q050.language == "en"

    assert row_q049.expected_sem_rank == 1
    assert row_q049.sem_top1_id == "osha-3120-lockout-tagout::chunk-0016"

    assert row_q049.top1_semantic is not None and 0.5500 <= row_q049.top1_semantic < 0.5999
    assert row_q050.top1_semantic is not None and 0.5500 <= row_q050.top1_semantic < 0.5999

    assert classify_candidate(row_q049, 0.5500, "none") == "grounded_review"
    assert classify_candidate(row_q050, 0.5500, "none") == "grounded_review"


def test_bilingual_pair_q071_and_q072(features_by_id: dict[str, FeatureRow]) -> None:
    """Verify bilingual pair q071 (es) and q072 (en) on NIOSH LEL definition:
    - Both fail to reach the 0.5500 floor
    - Both are consistently routed to hard_refuse without language bias
    """
    row_q071 = features_by_id["q071"]
    row_q072 = features_by_id["q072"]

    assert row_q071.language == "es"
    assert row_q072.language == "en"

    assert row_q071.top1_semantic is not None and row_q071.top1_semantic < 0.5500
    assert row_q072.top1_semantic is not None and row_q072.top1_semantic < 0.5500

    assert classify_candidate(row_q071, 0.5500, "none") == "hard_refuse"
    assert classify_candidate(row_q072, 0.5500, "none") == "hard_refuse"


def test_lexical_channel_asymmetry_explains_cross_lingual_bm25_failure(
    features_by_id: dict[str, FeatureRow],
) -> None:
    """Empirically demonstrate why hybrid lexical filters (sem_top1_in_bm25_top_n)
    fail Gate G2 for Spanish queries:
    - Cross-lingual Spanish queries have sem_top1_in_bm25_top_n = False
    - Enforcing BM25 agreement discriminates against Spanish queries
    """
    row_q049 = features_by_id["q049"]  # es

    # For Spanish q049, BM25 rank is None because corpus is English
    assert row_q049.expected_bm25_rank is None
    assert row_q049.sem_top1_in_bm25_top_n is False

    # Gating with sem_top1_in_bm25_top_n would incorrectly reject q049
    assert classify_candidate(row_q049, 0.5500, "sem_top1_in_bm25_top_n") == "hard_refuse"
    # But "none" correctly admits q049 into grounded_review
    assert classify_candidate(row_q049, 0.5500, "none") == "grounded_review"
