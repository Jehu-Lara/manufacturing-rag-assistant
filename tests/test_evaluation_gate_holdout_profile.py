from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.models import RetrievalResult
from src.features.evaluation import gate_holdout_profile


def _result(score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c1",
        fused_score=1.0,
        semantic_rank=1,
        semantic_score=score,
        bm25_rank=1,
        bm25_score=1.0,
        metadata={"chunk_id": "c1"},
    )


class _ScriptedRetriever:
    """Returns a top1-semantic score keyed by question id embedded in the text."""

    def __init__(self, score_by_id: dict[str, float]) -> None:
        self._score_by_id = score_by_id

    def retrieve(self, query_text: str, k: int = 5, top_n: int = 20) -> list[RetrievalResult]:
        qid = query_text.split("::", 1)[0]
        return [_result(self._score_by_id[qid])]


def _holdout(tmp_path: Path, score_by_id: dict[str, float]) -> Path:
    questions = []
    ids: list[str] = []
    for lang in ("en", "es"):
        for answerable in (True, False):
            prefix = "a" if answerable else "u"
            for n in range(12):
                qid = f"{prefix}{lang}{n}"
                ids.append(qid)
                questions.append(
                    {
                        "id": qid,
                        "pair_id": f"{prefix}{n}",
                        "question": f"{qid}:: some borderline question",
                        "language": lang,
                        "answerable": answerable,
                    }
                )
    assert len(questions) == 48
    path = tmp_path / "gate_holdout.json"
    path.write_text(
        json.dumps({"version": "1.0.0", "status": "draft", "questions": questions}), encoding="utf-8"
    )
    return path


def _all_grey() -> dict[str, float]:
    scores = {}
    for lang in ("en", "es"):
        for prefix in ("a", "u"):
            for n in range(12):
                scores[f"{prefix}{lang}{n}"] = 0.57
    return scores


def test_profile_all_grey_passes(tmp_path: Path) -> None:
    path = _holdout(tmp_path, _all_grey())
    report_path = gate_holdout_profile.run(
        holdout_path=path, retriever=_ScriptedRetriever(_all_grey()), report_dir=tmp_path
    )
    assert report_path.exists()
    profile = json.loads((tmp_path / "gate_holdout_band_profile.json").read_text(encoding="utf-8"))
    assert all(cell["grounded_review"] == 12 for cell in profile["cells"])
    assert b"\r\n" not in report_path.read_bytes()
    assert b"\r\n" not in (tmp_path / "gate_holdout_band_profile.json").read_bytes()
    assert not report_path.read_bytes().endswith(b"\n\n")


def test_profile_fails_when_a_cell_has_too_few_grey(tmp_path: Path) -> None:
    scores = _all_grey()
    # knock the en/answerable cell down to 2 grey (10 hard-refused)
    for n in range(10):
        scores[f"aen{n}"] = 0.10
    path = _holdout(tmp_path, scores)
    with pytest.raises(SystemExit):
        gate_holdout_profile.run(
            holdout_path=path, retriever=_ScriptedRetriever(scores), report_dir=tmp_path
        )


def test_profile_buckets_by_band(tmp_path: Path) -> None:
    scores = _all_grey()
    scores["aen0"] = 0.20  # hard_refuse
    scores["aen1"] = 0.95  # confident
    path = _holdout(tmp_path, scores)
    status, cells = gate_holdout_profile.profile_holdout(
        holdout_path=path, retriever=_ScriptedRetriever(scores)
    )
    assert status == "draft"
    en_answerable = next(c for c in cells if c.language == "en" and c.answerable)
    assert (en_answerable.hard_refuse, en_answerable.grounded_review, en_answerable.confident) == (1, 10, 1)


def test_profile_rejects_holdout_without_48_questions(tmp_path: Path) -> None:
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"status": "draft", "questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="48 questions"):
        gate_holdout_profile.profile_holdout(holdout_path=path, retriever=_ScriptedRetriever({}))
