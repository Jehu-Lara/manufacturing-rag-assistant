from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.features.evaluation import failure_classification as fc

_COMMITTED_DETAILS = (
    Path(__file__).resolve().parent.parent
    / "eval"
    / "reports"
    / "retrieval_details_v1.1.0__raw-v1__off.jsonl"
)


def _entry(chunk_id: str, rank: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "semantic_score": 0.5,
        "semantic_rank": rank,
        "bm25_rank": rank,
        "fused_score": 1.0 / rank,
    }


def _top5(*chunk_ids: str) -> list[dict[str, Any]]:
    return [_entry(cid, i) for i, cid in enumerate(chunk_ids, start=1)]


def test_gate_over_refusal_expected_chunk_in_top5_gate_refuses() -> None:
    top5 = _top5("doc-a::chunk-0003", "doc-a::chunk-0001", "doc-b::chunk-0000")
    assert (
        fc.classify_failure(["doc-a::chunk-0003"], "doc-a", top5, gate_confident=False)
        == "gate-over-refusal"
    )


def test_same_document_decoy_expected_chunk_absent_top1_same_doc() -> None:
    top5 = _top5("doc-a::chunk-0001", "doc-a::chunk-0002", "doc-c::chunk-0000")
    assert (
        fc.classify_failure(["doc-a::chunk-0018"], "doc-a", top5, gate_confident=True)
        == "same-document-decoy"
    )


def test_cross_document_decoy_expected_doc_present_but_not_top1() -> None:
    top5 = _top5("doc-b::chunk-0021", "doc-a::chunk-0005", "doc-c::chunk-0000")
    assert (
        fc.classify_failure(["doc-a::chunk-0016"], "doc-a", top5, gate_confident=False)
        == "cross-document-decoy"
    )


def test_retrieval_miss_expected_document_absent_from_top5() -> None:
    top5 = _top5("doc-b::chunk-0001", "doc-c::chunk-0002", "doc-d::chunk-0003")
    assert (
        fc.classify_failure(["doc-a::chunk-0007"], "doc-a", top5, gate_confident=True)
        == "retrieval-miss"
    )


def test_gate_over_refusal_takes_precedence_over_decoy_shape() -> None:
    # expected chunk present at rank 4, but top-1 is a sibling decoy: still gate-over-refusal
    top5 = _top5("doc-a::chunk-0001", "doc-a::chunk-0002", "doc-a::chunk-0009", "doc-a::chunk-0018")
    assert (
        fc.classify_failure(["doc-a::chunk-0018"], "doc-a", top5, gate_confident=False)
        == "gate-over-refusal"
    )


def test_is_failure_and_success_discrimination() -> None:
    success = {
        "top5": _top5("doc-a::chunk-0003", "doc-a::chunk-0001"),
        "expected_chunk_ids": ["doc-a::chunk-0003"],
        "gate_decision": "answer",
    }
    gate_fail = {**success, "gate_decision": "refuse"}
    recall_fail = {
        "top5": _top5("doc-a::chunk-0001"),
        "expected_chunk_ids": ["doc-a::chunk-0003"],
        "gate_decision": "answer",
    }
    assert fc.is_failure(success) is False
    assert fc.is_failure(gate_fail) is True
    assert fc.is_failure(recall_fail) is True


def test_run_over_synthetic_jsonl_counts_and_report(tmp_path: Path) -> None:
    import json

    records = [
        # success — not classified
        {
            "id": "q1",
            "lang": "en",
            "top5": _top5("doc-a::chunk-0003"),
            "gate_decision": "answer",
            "expected_document_id": "doc-a",
            "expected_chunk_ids": ["doc-a::chunk-0003"],
        },
        # gate-over-refusal
        {
            "id": "q2",
            "lang": "es",
            "top5": _top5("doc-a::chunk-0003"),
            "gate_decision": "refuse",
            "expected_document_id": "doc-a",
            "expected_chunk_ids": ["doc-a::chunk-0003"],
        },
        # same-document-decoy
        {
            "id": "q3",
            "lang": "en",
            "top5": _top5("doc-a::chunk-0001", "doc-a::chunk-0002"),
            "gate_decision": "answer",
            "expected_document_id": "doc-a",
            "expected_chunk_ids": ["doc-a::chunk-0018"],
        },
    ]
    details = tmp_path / "retrieval_details_v1.1.0__raw-v1__off.jsonl"
    details.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out = tmp_path / "classification.md"

    out_path, counts = fc.run(details, out)

    assert out_path == out
    assert counts["gate-over-refusal"] == 1
    assert counts["same-document-decoy"] == 1
    assert counts["cross-document-decoy"] == 0
    assert counts["retrieval-miss"] == 0
    text = out.read_text(encoding="utf-8")
    assert "# Failure classification" in text
    assert "q2" in text and "q3" in text
    assert "q1" not in text


def test_report_header_reflects_the_runs_real_profile_and_mode(tmp_path: Path) -> None:
    import json

    records = [
        {
            "id": "q2",
            "lang": "es",
            "top5": _top5("doc-a::chunk-0003"),
            "gate_decision": "refuse",
            "expected_document_id": "doc-a",
            "expected_chunk_ids": ["doc-a::chunk-0003"],
        },
    ]
    details = tmp_path / "retrieval_details_v1.1.0__contextual-v1__semantic.jsonl"
    details.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out = tmp_path / "classification.md"

    fc.run(details, out)

    text = out.read_text(encoding="utf-8")
    assert (
        "# Failure classification — eval_set v1.1.0, contextual-v1 index, expansion_mode=semantic"
        in text
    )
    assert "raw-v1 index" not in text
    assert "Phase 2 Task 3" not in text


def test_profile_mode_from_details_name_raises_on_unrecognized() -> None:
    with pytest.raises(ValueError):
        fc._profile_mode_from_details_name("classification.md")
    assert (
        fc._profile_mode_from_details_name(
            "retrieval_details_v1.1.0__contextual-v1__both.jsonl"
        )
        == ("contextual-v1", "both")
    )


def test_run_derives_out_path_from_details_name_not_raw_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.setattr(fc, "REPORT_DIR", tmp_path)
    records = [
        {
            "id": "q2",
            "lang": "es",
            "top5": _top5("doc-a::chunk-0003"),
            "gate_decision": "refuse",
            "expected_document_id": "doc-a",
            "expected_chunk_ids": ["doc-a::chunk-0003"],
        },
    ]
    details = tmp_path / "retrieval_details_v1.1.0__contextual-v1__off.jsonl"
    details.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    out_path, _ = fc.run(details)

    assert out_path == tmp_path / "classification_v1.1.0__contextual-v1__off.md"
    assert out_path.name != "classification_v1.1.0__raw-v1__off.md"
    assert out_path.exists()


def test_committed_jsonl_still_yields_the_frozen_counts() -> None:
    """CI guard: if the committed retrieval-details JSONL is ever regenerated,
    the failure-class counts must still be 15/9/2/0 (the frozen Phase-1 ruling).
    """
    if not _COMMITTED_DETAILS.exists():
        pytest.skip("committed retrieval_details JSONL not present in this checkout")

    records = fc.load_details(_COMMITTED_DETAILS)
    counts = fc.count_classes(fc.classify_details(records))

    assert counts == fc.EXPECTED_COUNTS == {
        "gate-over-refusal": 15,
        "same-document-decoy": 9,
        "cross-document-decoy": 2,
        "retrieval-miss": 0,
    }
