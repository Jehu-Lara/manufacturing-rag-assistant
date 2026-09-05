from __future__ import annotations

import pytest

from src.domain.models import ChunkMetadata
from src.domain.policies import RRF_K, fuse_rankings
from src.features.evaluation import ablation_eval, fusion_sweep
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K, HybridRetriever


def _metadata(chunk_id: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        document_id="doc",
        document_title="Doc",
        revision="1.0",
        section_heading="Section",
        source_type="public",
        source_url_or_note="note",
        source_page_range=None,
        md_line_range="1-2",
        chunk_token_count=3,
        chunk_text=f"text for {chunk_id}",
    )


class _FakeVectorStore:
    def __init__(self, ranked: list[str]) -> None:
        self._ranked = ranked

    def query(self, text: str, top_n: int) -> list[tuple[str, float, ChunkMetadata]]:
        return [(cid, 1.0 - i / 100, _metadata(cid)) for i, cid in enumerate(self._ranked[:top_n])]

    def get_metadata(self, chunk_id: str) -> ChunkMetadata:
        return _metadata(chunk_id)


class _FakeLexicalIndex:
    def __init__(self, ranked: list[str]) -> None:
        self._ranked = ranked

    def build_index(self, chunks, **kwargs) -> None:
        return None

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        return [(cid, 10.0 - i) for i, cid in enumerate(self._ranked[:top_n])]


SEMANTIC = [f"c{i:02d}" for i in range(20)]
LEXICAL = [f"c{i:02d}" for i in (19, 5, 3, 18, 17, 2, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 4, 1, 0)]


def test_baseline_rule_reproduces_the_shipped_retriever_exactly() -> None:
    """The sweep replays fusion offline instead of re-running the retriever, so
    the baseline rule must be the REAL fuse_rankings, not a lookalike. If this
    ever diverges, every other row in the sweep is measured against a fiction."""
    retriever = HybridRetriever(_FakeVectorStore(SEMANTIC), _FakeLexicalIndex(LEXICAL))
    served = [r.chunk_id for r in retriever.retrieve("q", k=SEMANTIC_EXTRACTION_K)]

    replayed = [cid for cid, _ in fusion_sweep.RULES[fusion_sweep.BASELINE_RULE](SEMANTIC, LEXICAL)]

    assert replayed == served


def test_weighted_rrf_at_unit_weights_is_plain_rrf() -> None:
    """The weighted variant has to contain the shipped policy as its identity
    case, or a weight sweep measures two changes at once."""
    weighted = fusion_sweep.weighted_rrf(SEMANTIC, LEXICAL, k=RRF_K, semantic_weight=1.0, lexical_weight=1.0)

    assert weighted == fuse_rankings(SEMANTIC, LEXICAL)


def test_zero_lexical_weight_drops_lexical_only_candidates() -> None:
    """Weight 0 must remove a lexical-only chunk from the ranking entirely, not
    leave it at score 0 where the chunk_id tie-break could still surface it."""
    ranked = fusion_sweep.weighted_rrf(["a", "b"], ["z"], k=RRF_K, semantic_weight=1.0, lexical_weight=0.0)

    assert [cid for cid, _ in ranked] == ["a", "b"]


def test_every_rule_is_a_pure_function_of_the_two_id_lists() -> None:
    for name, rule in fusion_sweep.RULES.items():
        first = rule(SEMANTIC, LEXICAL)
        second = rule(SEMANTIC, LEXICAL)
        assert first == second, name


def test_rules_include_the_shipped_setting_and_at_least_one_of_each_lever() -> None:
    """The sweep exists to separate two levers: k (how flat the rank curve is)
    and channel weight (how much a both-channels bonus is worth)."""
    assert fusion_sweep.BASELINE_RULE in fusion_sweep.RULES
    assert any(name.startswith("rrf_k") and name != fusion_sweep.BASELINE_RULE for name in fusion_sweep.RULES)
    assert any("sem_x" in name for name in fusion_sweep.RULES)


def test_score_rule_reuses_the_ablation_arm_result() -> None:
    """One definition of Recall@5/MRR/per-language, one definition of the gates.
    A second copy would drift from the ablation it is meant to extend."""
    channels = {
        "q1": fusion_sweep.QuestionChannels(
            qid="q1", language="en", expected_chunk_ids=["c00"], semantic_ids=SEMANTIC, lexical_ids=LEXICAL
        )
    }

    result = fusion_sweep.score_rule(fusion_sweep.BASELINE_RULE, channels)

    assert isinstance(result, ablation_eval.ArmResult)
    assert result.name == fusion_sweep.BASELINE_RULE
    assert set(result.hits) == {"q1"}


def test_report_states_that_k60_is_a_byte_stable_invariant_this_run_did_not_change() -> None:
    """RRF k=60 is pinned in CLAUDE.md. A sweep that reads as a recommendation
    invites someone to edit the constant off the back of 80 questions."""
    channels = {
        "q1": fusion_sweep.QuestionChannels(
            qid="q1", language="en", expected_chunk_ids=["c00"], semantic_ids=SEMANTIC, lexical_ids=LEXICAL
        )
    }
    results = {name: fusion_sweep.score_rule(name, channels) for name in fusion_sweep.RULES}

    report = fusion_sweep.render_report(results, "1.1.0", "contextual-v1")

    assert "byte-stable" in report
    assert "measurement, not a decision" in report


def test_report_names_the_questions_each_rule_moves() -> None:
    channels = {
        "q1": fusion_sweep.QuestionChannels(
            qid="q1", language="en", expected_chunk_ids=["c19"], semantic_ids=SEMANTIC, lexical_ids=LEXICAL
        ),
        "q2": fusion_sweep.QuestionChannels(
            qid="q2", language="es", expected_chunk_ids=["c00"], semantic_ids=SEMANTIC, lexical_ids=LEXICAL
        ),
    }
    results = {name: fusion_sweep.score_rule(name, channels) for name in fusion_sweep.RULES}

    report = fusion_sweep.render_report(results, "1.1.0", "contextual-v1")

    assert "New misses" in report
    assert "Rescues" in report


def test_unknown_rule_raises() -> None:
    with pytest.raises(KeyError):
        fusion_sweep.score_rule("rrf_k9000", {})
