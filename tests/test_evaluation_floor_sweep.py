from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.config import DEFAULT_REFUSAL_COSINE_THRESHOLD, DEFAULT_REFUSAL_REVIEW_FLOOR
from src.domain.models import RetrievalResult
from src.features.evaluation import floor_sweep as fs
from src.features.retrieval import index_manifest


def _rr(
    chunk_id: str,
    *,
    sem_rank: int | None,
    sem_score: float | None,
    bm25_rank: int | None,
    bm25_score: float | None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        fused_score=0.0,
        semantic_rank=sem_rank,
        semantic_score=sem_score,
        bm25_rank=bm25_rank,
        bm25_score=bm25_score,
        metadata={},
    )


def _row(score: float | None, **overrides: object) -> fs.FeatureRow:
    values: dict[str, object] = {
        "question_id": "q1",
        "language": "en",
        "answerable": True,
        "cohort": "eval_set",
        "top1_semantic": score,
        "sem_margin": 0.05 if score is not None else None,
        "sem_top1_id": "a" if score is not None else None,
        "bm25_top1_id": "a",
        "expected_sem_rank": 1,
        "expected_bm25_rank": 1,
        "sem_bm25_top1_agree": score is not None,
        "sem_top1_in_bm25_top_n": score is not None,
        "channels_overlap_top_n": score is not None,
    }
    values.update(overrides)
    return fs.FeatureRow(**values)  # type: ignore[arg-type]


def _all_pinned_rows() -> list[fs.FeatureRow]:
    scores = {"r001": 0.5642, "r002": 0.5656, "r018": 0.5630, "r019": 0.5001, "r020": 0.5420}
    return [
        _row(
            score,
            question_id=question_id,
            language="es" if question_id in {"r002", "r019"} else "en",
            answerable=question_id in {"r001", "r002"},
            cohort="regression_controls",
        )
        for question_id, score in scores.items()
    ]


def _manifest() -> index_manifest.IndexManifest:
    return index_manifest.IndexManifest(
        index_profile="contextual-v1",
        chunks_sha256="a" * 64,
        corpus_sha256="b" * 64,
        embedding_model="BAAI/bge-m3",
        embedding_revision="revision",
        build_commit="c" * 40,
        chunk_count=228,
    )


def test_constants_are_immutable_and_sourced_from_config() -> None:
    assert fs.SIGNAL_TOP_N < fs.CHANNEL_TOP_N <= fs.RETRIEVE_K
    assert fs.THRESHOLD == DEFAULT_REFUSAL_COSINE_THRESHOLD == 0.5999
    assert fs.CURRENT_FLOOR == DEFAULT_REFUSAL_REVIEW_FLOOR == 0.5500
    assert isinstance(fs.FLOORS, tuple)
    assert isinstance(fs.SIGNAL_NAMES, tuple)


def test_extract_features_uses_best_expected_rank_and_channel_signals() -> None:
    results = [
        _rr("gold-b", sem_rank=1, sem_score=0.57, bm25_rank=3, bm25_score=4.0),
        _rr("other", sem_rank=2, sem_score=0.50, bm25_rank=1, bm25_score=6.0),
        _rr("gold-a", sem_rank=3, sem_score=0.49, bm25_rank=2, bm25_score=5.0),
    ]
    row = fs.extract_features(
        results,
        question_id="r001",
        language="en",
        answerable=True,
        cohort="regression_controls",
        expected_chunk_ids=["gold-a", "gold-b"],
    )
    assert row.top1_semantic == pytest.approx(0.57)
    assert row.sem_margin == pytest.approx(0.07)
    assert row.expected_sem_rank == 1
    assert row.expected_bm25_rank == 2
    assert row.sem_top1_in_bm25_top_n is True
    assert row.channels_overlap_top_n is True
    assert row.sem_bm25_top1_agree is False


def test_extract_features_handles_empty_results_like_serving() -> None:
    row = fs.extract_features(
        [],
        question_id="empty",
        language="es",
        answerable=False,
        cohort="eval_set",
        expected_chunk_ids=[],
    )
    assert row.top1_semantic is None
    assert row.sem_top1_id is None
    assert fs.classify_candidate(row, 0.50, "none") == "hard_refuse"


@pytest.mark.parametrize("score", [None, 0.40, 0.5171, 0.5499, 0.55, 0.5998, 0.5999, 0.72])
def test_current_candidate_matches_real_refusal_policy(score: float | None) -> None:
    from src.domain.policies import RefusalPolicy

    policy = RefusalPolicy(fs.THRESHOLD, mode="grounded_review", review_floor=fs.CURRENT_FLOOR)
    assert fs.classify_candidate(_row(score), *fs.CURRENT_RULE) == policy.classify_score(score)


def test_unknown_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown entry signal"):
        fs.classify_candidate(_row(0.53), 0.50, "typo")


def test_confident_wins_even_if_candidate_floor_is_higher() -> None:
    assert fs.classify_candidate(_row(0.70), 0.80, "none") == "confident"


def test_collect_rows_rejects_non_contextual_profile_before_loading_data() -> None:
    with pytest.raises(RuntimeError, match="requires the coherent contextual-v1/off index"):
        fs.collect_rows("raw-v1")


def test_signal_can_only_gate_the_review_band() -> None:
    passing = _row(0.53, sem_bm25_top1_agree=True)
    failing = _row(0.53, sem_bm25_top1_agree=False)
    assert fs.classify_candidate(passing, 0.50, "sem_bm25_top1_agree") == "grounded_review"
    assert fs.classify_candidate(failing, 0.50, "sem_bm25_top1_agree") == "hard_refuse"
    assert fs.classify_candidate(failing, 0.50, "none") == "grounded_review"
    assert fs.classify_candidate(_row(0.70), 0.50, "sem_bm25_top1_agree") == "confident"


def test_grid_counts_partition_every_slice() -> None:
    rows = [_row(0.52), _row(0.58, question_id="q2"), _row(0.40, question_id="u", answerable=False)]
    stats = fs.sweep_grid(rows)
    assert {(stat.floor, stat.signal) for stat in stats} == {
        (floor, signal) for floor in fs.FLOORS for signal in fs.SIGNAL_NAMES
    }
    assert all(stat.hard_refuse + stat.grounded_review + stat.confident == stat.n for stat in stats)


def test_g3_rejects_too_many_new_unanswerable_reviews() -> None:
    rows = [
        _row(0.52, question_id=f"u{i}", language="es", answerable=False)
        for i in range(3)
    ]
    assessment = fs.assess_rule(rows, fs.sweep_grid(rows), 0.50, "none")
    assert assessment.g3_unanswerable_containment is False


def test_g4_requires_every_pinned_control() -> None:
    rows = _all_pinned_rows()[:-1]
    assessment = fs.assess_rule(rows, fs.sweep_grid(rows), *fs.CURRENT_RULE)
    assert assessment.g4_regression_controls is False


def test_g4_accepts_complete_current_control_baseline() -> None:
    rows = _all_pinned_rows()
    assessment = fs.assess_rule(rows, fs.sweep_grid(rows), *fs.CURRENT_RULE)
    assert assessment.g4_regression_controls is True


def test_write_run_is_atomic_and_confined_to_injected_directory(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "calibration"
    rows = [
        _row(0.52, question_id="answerable", language="es"),
        _row(0.40, question_id="unanswerable", answerable=False),
        *_all_pinned_rows(),
    ]
    run_dir = fs.write_run(
        rows,
        _manifest(),
        calibration_dir=calibration_dir,
        now=datetime(2026, 9, 2, tzinfo=UTC),
    )
    assert run_dir.parent == calibration_dir.resolve()
    assert {path.name for path in run_dir.iterdir()} == {
        "features.jsonl",
        "grid.json",
        "report.md",
        "sanitized_summary.md",
    }
    assert not list(calibration_dir.glob("*.partial"))
    summary = (run_dir / "sanitized_summary.md").read_text(encoding="utf-8")
    assert "NOT APPLIED" in summary
    assert "answerable" not in summary


def test_write_run_cleans_partial_directory_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_report(*args: object, **kwargs: object) -> str:
        raise RuntimeError("report failure")

    monkeypatch.setattr(fs, "render_report", fail_report)
    with pytest.raises(RuntimeError, match="report failure"):
        fs.write_run(
            [_row(0.52), *_all_pinned_rows()],
            _manifest(),
            calibration_dir=tmp_path,
            now=datetime(2026, 9, 2, tzinfo=UTC),
        )
    assert not list(tmp_path.iterdir())


def test_no_holdout_reference_in_serving_or_floor_sweep() -> None:
    repo = Path(__file__).resolve().parents[1]
    files = [repo / "src" / "main.py", repo / "src" / "features" / "evaluation" / "floor_sweep.py"]
    for relative in ("domain", "adapters", "features/query", "core"):
        files.extend((repo / "src" / relative).rglob("*.py"))
    offenders: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "gate_holdout_v" in node.value:
                    offenders.append(f"{path}: literal")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", "") or "", *(alias.name for alias in node.names)]
                if any("gate_holdout" in name for name in names):
                    offenders.append(f"{path}: import")
    assert not offenders, offenders
