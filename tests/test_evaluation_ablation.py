from __future__ import annotations

import pytest

from src.adapters.secondary.lexical.null_lexical_index import NullLexicalIndex
from src.domain.ports import LexicalIndexPort
from src.features.evaluation import ablation_eval


def test_null_index_satisfies_the_port() -> None:
    assert isinstance(NullLexicalIndex(), LexicalIndexPort)


def test_null_index_returns_no_hits() -> None:
    assert NullLexicalIndex().query("anything at all", 20) == []


def test_semantic_only_arm_uses_the_null_lexical_channel(monkeypatch) -> None:
    """The ablation must run the real fusion code with an EMPTY BM25 ranking,
    not a special-cased branch that skips fusion — otherwise it measures a code
    path production never takes."""

    class _Store:
        def query(self, text, top_n):
            return []

        def get_metadata(self, chunk_id):
            raise NotImplementedError

    monkeypatch.setattr(
        ablation_eval, "build_retriever", lambda *a, **k: ablation_eval.HybridRetriever(_Store(), object())
    )

    retriever = ablation_eval.build_arm("semantic_only")

    assert isinstance(retriever._lexical_index, NullLexicalIndex)


def test_baseline_arm_is_the_untouched_retriever(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(ablation_eval, "build_retriever", lambda *a, **k: sentinel)

    assert ablation_eval.build_arm(ablation_eval.BASELINE_ARM) is sentinel


def test_unknown_arm_raises() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        ablation_eval.build_arm("semantic_and_vibes")


def _arm(name: str, hits: dict[str, bool], languages: dict[str, str] | None = None) -> ablation_eval.ArmResult:
    return ablation_eval.ArmResult(
        name=name,
        hits=dict(hits),
        language_by_id=dict(languages or {q: "en" for q in hits}),
        reciprocal_ranks={q: 1.0 if hit else 0.0 for q, hit in hits.items()},
    )


def test_compare_reports_new_misses_not_just_aggregate_recall() -> None:
    """Equal recall can hide a swap: one question won, another lost. The
    acceptance gate is zero NEW misses, so the comparison must name them."""
    baseline = _arm("b", {"q1": True, "q2": True, "q3": False})
    candidate = _arm("c", {"q1": True, "q2": False, "q3": True})

    delta = ablation_eval.compare(baseline, candidate)

    assert delta.new_misses == ["q2"]
    assert delta.rescues == ["q3"]
    assert delta.recall_delta == 0.0


def test_recall_is_reported_per_language() -> None:
    result = _arm(
        "x",
        {"q1": True, "q2": False, "q3": True, "q4": True},
        {"q1": "en", "q2": "en", "q3": "es", "q4": "es"},
    )

    assert result.recall_for("en") == 0.5
    assert result.recall_for("es") == 1.0
    assert result.recall_at_5 == 0.75


def test_gates_are_the_spec_thresholds() -> None:
    """These four numbers are the audit's acceptance criteria; a candidate that
    misses any of them does not replace contextual-v1/off."""
    names = [name for name, _ in ablation_eval.GATES]

    assert names == [
        "EN Recall@5 >= 0.917",
        "ES Recall@5 >= 0.844",
        "Global Recall@5 >= 0.887",
    ]


def test_snowball_arm_fails_closed_without_the_experiment_dependency(monkeypatch) -> None:
    """The Snowball arm needs an experiment-only dependency that is deliberately
    absent from the runtime lock. It must say so, not fall back silently."""
    monkeypatch.setattr(ablation_eval, "build_retriever", lambda *a, **k: object())
    monkeypatch.setattr(
        ablation_eval,
        "_snowball_index",
        lambda: (_ for _ in ()).throw(RuntimeError("requirements-experiments.txt")),
    )

    with pytest.raises(RuntimeError, match="requirements-experiments"):
        ablation_eval.build_arm("hybrid_snowball_bilingual")


def test_report_names_the_questions_bm25_uniquely_rescues() -> None:
    """When the lexical channel wins a question outright, the report has to say
    so by name — that is what refuting 'BM25 aporta ~0' looks like."""
    results = {
        ablation_eval.BASELINE_ARM: _arm(ablation_eval.BASELINE_ARM, {"q1": True, "q2": True}),
        "semantic_only": _arm("semantic_only", {"q1": True, "q2": False}),
    }

    report = ablation_eval.render_report(results, "1.1.0", "contextual-v1")

    assert "q2" in report
    assert "refuted in its favour" in report


def test_report_calls_a_net_negative_lexical_channel_what_it_is() -> None:
    """The opposite outcome must be stated just as plainly: no exclusive rescue
    AND questions lost to fusion is not 'contributes ~nothing', it is harmful —
    and the report must still frame it as a measurement, not a decision."""
    results = {
        ablation_eval.BASELINE_ARM: _arm(ablation_eval.BASELINE_ARM, {"q1": True, "q2": False}),
        "semantic_only": _arm("semantic_only", {"q1": True, "q2": True}),
    }

    report = ablation_eval.render_report(results, "1.1.0", "contextual-v1")

    assert "net-negative" in report
    assert "q2" in report
    assert "measurement, not a decision" in report


def test_the_shipped_baseline_passes_its_own_gates() -> None:
    """A gate the current production configuration fails is a broken gate, not
    a regression. EN is 44/48 = 0.91666... and ES is 27/32 = 0.84375; the
    audit's thresholds are those figures rounded to three decimals, so the
    comparison has to round the same way."""
    baseline = ablation_eval.ArmResult(
        name=ablation_eval.BASELINE_ARM,
        hits={f"en{i}": i < 44 for i in range(48)} | {f"es{i}": i < 27 for i in range(32)},
        language_by_id={f"en{i}": "en" for i in range(48)} | {f"es{i}": "es" for i in range(32)},
    )

    assert all(check(baseline) for _, check in ablation_eval.GATES)
