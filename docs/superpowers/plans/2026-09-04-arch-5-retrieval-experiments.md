# Bucket 5 — Retrieval Experiments (Ablation + Reranker), Default-Off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two unproven beliefs with measurements. First, settle whether BM25 contributes anything, by running a real ablation (semantic-only / current `word-lower-v1` BM25 / bilingual Snowball BM25). Then, if and only if the evidence justifies it, add a `RerankerPort` with `bge-reranker-v2-m3` over the first 20 fused results — permuting only that window, preserving the tail and the semantic top-1 signal the refusal gate depends on. Both ship default-off and neither changes what is served.

**Architecture:** Three ordered tasks. T1 builds the ablation harness and a null lexical channel, and runs it — its output is evidence, not a code change to serving. T2 adds the alternative Snowball tokenizer as a **separate** adapter writing to a **separate** index path, so the live index is never touched, and adds it as a third ablation arm. T3 adds the reranker port, adapter and opt-in wiring, with the gate-preservation property pinned by test. T1 → T2 → T3, and **T3 does not start until the owner has read T1/T2's report and said to proceed.**

**Tech Stack:** Python 3.11, rank-bm25, `snowballstemmer` (experiment-only, never in the runtime lock), `sentence_transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-architecture-remediation-design.md`

## Global Constraints

- Execution is gated: PR #9 resolved by the owner, then a new branch cut from `master` with the owner's authorization.
- **Nothing here changes a default.** `expansion_mode` stays `off`, the served profile stays `contextual-v1`, `REFUSAL_POLICY` stays `binary`, the reranker defaults to `None`. Serving behaviour is byte-identical unless the owner separately approves a flip.
- **Acceptance gates for any candidate replacing `contextual-v1/off`** — all four must hold, and even then the flip is a separate owner decision:
  - EN Recall@5 ≥ 0.917
  - ES Recall@5 ≥ 0.844
  - Global Recall@5 ≥ 0.887
  - **Zero new misses** relative to the current baseline (a candidate that trades a won question for a different won question fails, even at equal recall)
- **No new runtime dependency.** `snowballstemmer` and the reranker model go in `requirements-experiments.txt`, which is never installed into the image and never enters `requirements-lock.txt`. If the owner later ships either, that is its own task with a lock regenerated in a Debian Bookworm amd64 / Python 3.11 container — never a local Windows `pip freeze`.
- **No reindex of the live index.** Every experimental index writes to its own path under `retrieval/output/experiments/`, which must be gitignored.
- Frozen datasets are never re-stamped. Reports go to new profile-labelled filenames; `eval/reports/*_v1.*` and `*__raw-v1__off.*` are never overwritten.
- `0.5500` is never recalibrated, and never against the Phase 3 holdout.
- After each task: `pytest tests/test_evaluation_ablation.py tests/test_hybrid_retriever_use_case.py -q` green. End of bucket: `pytest -q`, `ruff check src tests`, `mypy src` green.

---

## File Structure

- `src/features/evaluation/ablation_eval.py` — **new.** The harness: builds one retriever per arm, runs the eval set and the regression queries through each, writes a comparison report with per-question win/loss deltas.
- `src/adapters/secondary/lexical/null_lexical_index.py` — **new.** `LexicalIndexPort` returning `[]`. This is how the semantic-only arm is expressed without special-casing `HybridRetriever`.
- `src/adapters/secondary/lexical/snowball_bm25_index.py` — **new.** Subclasses/wraps `Bm25LexicalIndex` with a bilingual EN+ES Snowball tokenizer and `LEXICAL_PROFILE = "snowball-bilingual-v1"`.
- `src/domain/ports.py` — gains `RerankerPort`.
- `src/adapters/secondary/reranker/flag_reranker.py` — **new.** `CrossEncoder`-backed `RerankerPort`.
- `src/features/retrieval/use_cases.py` — `HybridRetriever(..., reranker: RerankerPort | None = None, rerank_window: int = 20)`.
- `requirements-experiments.txt` — **new**, unpinned-by-hash, never installed in the image.
- `docs/eval/ablation_summary.md` — the committed, sanitized result.
- Tests: `tests/test_evaluation_ablation.py`, `tests/test_null_lexical_index.py`, `tests/test_snowball_bm25_index.py`, `tests/test_reranker_adapter.py`, `tests/test_hybrid_retriever_use_case.py`.

---

### Task 1: Ablation harness — does BM25 contribute anything?

**Files:**
- Create: `src/adapters/secondary/lexical/null_lexical_index.py`, `src/features/evaluation/ablation_eval.py`, `tests/test_null_lexical_index.py`, `tests/test_evaluation_ablation.py`
- Test: as above

**Interfaces:**
- Produces:
  - `NullLexicalIndex` implementing `LexicalIndexPort`: `build_index(chunks, **kwargs) -> None` (no-op) and `query(text, top_n) -> []`
  - `ablation_eval.ARMS: tuple[str, ...] = ("semantic_only", "hybrid_word_lower", "hybrid_snowball_bilingual")`
  - `ablation_eval.run(arms: Sequence[str] = ARMS, *, index_profile: IndexProfile = "contextual-v1") -> Path` — returns the written report path
  - `ablation_eval.compare(baseline: ArmResult, candidate: ArmResult) -> ArmDelta` with `ArmDelta.new_misses: list[str]` (question ids the baseline got and the candidate lost) and `.rescues: list[str]`
- Consumes: `src.features.evaluation._eval_retriever.build_retriever`, `src.features.evaluation.metrics.{recall_at_k, reciprocal_rank, mean_reciprocal_rank}`, the frozen `eval/eval_set.json` and `eval/regression_queries.json`.

**Why `NullLexicalIndex` and not a flag:** an `if lexical is None` branch inside `HybridRetriever` would put experiment-only code on the serving path. A port implementation returning an empty ranking exercises the *real* fusion code — `fuse_rankings(semantic_ids, [])` — which is what an honest ablation must measure.

- [ ] **Step 1: Write the failing tests** — create `tests/test_null_lexical_index.py`:

```python
from __future__ import annotations

from src.adapters.secondary.lexical.null_lexical_index import NullLexicalIndex
from src.domain.ports import LexicalIndexPort


def test_null_index_satisfies_the_port() -> None:
    assert isinstance(NullLexicalIndex(), LexicalIndexPort)


def test_null_index_returns_no_hits() -> None:
    assert NullLexicalIndex().query("anything at all", 20) == []
```

and `tests/test_evaluation_ablation.py`:

```python
from __future__ import annotations

from src.features.evaluation import ablation_eval


def test_semantic_only_arm_uses_the_null_lexical_channel() -> None:
    """The ablation must run the real fusion code with an empty BM25 ranking,
    not a special-cased branch that skips fusion entirely — otherwise it
    measures a code path production never takes."""
    from src.adapters.secondary.lexical.null_lexical_index import NullLexicalIndex

    retriever = ablation_eval.build_arm("semantic_only", verify_physical_coherence=False)

    assert isinstance(retriever._lexical_index, NullLexicalIndex)


def test_compare_reports_new_misses_not_just_aggregate_recall() -> None:
    """Equal recall can hide a swap: one question won, another lost. The
    acceptance gate is zero NEW misses, so the comparison must name them."""
    baseline = ablation_eval.ArmResult(name="b", hits={"q1": True, "q2": True, "q3": False})
    candidate = ablation_eval.ArmResult(name="c", hits={"q1": True, "q2": False, "q3": True})

    delta = ablation_eval.compare(baseline, candidate)

    assert delta.new_misses == ["q2"]
    assert delta.rescues == ["q3"]
    assert delta.recall_delta == 0.0


def test_unknown_arm_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown arm"):
        ablation_eval.build_arm("semantic_and_vibes")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_null_lexical_index.py tests/test_evaluation_ablation.py -q`

Expected: FAIL — neither module exists.

- [ ] **Step 3: Implement the null channel and the harness**

```python
# src/adapters/secondary/lexical/null_lexical_index.py
from __future__ import annotations

from typing import Any

from src.domain.models import ChunkMetadata


class NullLexicalIndex:
    """Implements LexicalIndexPort with no lexical signal at all. Exists so the
    ablation's semantic-only arm runs the real RRF fusion against an empty BM25
    ranking, rather than a branch that skips fusion — measuring a code path
    production never takes would answer the wrong question."""

    def build_index(self, chunks: list[ChunkMetadata], **kwargs: Any) -> None:
        return None

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        return []
```

`ablation_eval.py` holds `ArmResult` (`name`, `hits: dict[str, bool]`, `recall_at_5`, `mrr`, `per_language`), `ArmDelta`, `compare`, `build_arm`, `run`, and a `main()`. `build_arm` reuses `_eval_retriever.build_retriever` for the vector store and swaps the lexical channel:

```python
def build_arm(arm: str, *, verify_physical_coherence: bool = True) -> HybridRetriever:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}; expected one of {ARMS}")
    base = build_retriever(verify_physical_coherence=verify_physical_coherence)
    if arm == "hybrid_word_lower":
        return base
    lexical = NullLexicalIndex() if arm == "semantic_only" else _snowball_index()
    return HybridRetriever(base._vector_store, lexical, expansion_mode="off")
```

`_snowball_index()` raises a clear "run Task 2 first" `RuntimeError` until Task 2 lands. Reaching into `base._vector_store` is deliberate and belongs to offline evaluation only — add a one-line comment saying so, and do not add a public accessor to `HybridRetriever` for it.

The report must include, per arm and per language: Recall@5, MRR, and the explicit `new_misses` / `rescues` lists against the `hybrid_word_lower` baseline. Aggregate recall alone cannot satisfy the zero-new-misses gate.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_null_lexical_index.py tests/test_evaluation_ablation.py -q`

Expected: PASS (the Snowball arm test is deferred to Task 2).

- [ ] **Step 5: Run the two-arm ablation and write the result down**

Needs a coherent live `contextual-v1` index; it makes no LLM calls and costs nothing.

```bash
python -m src.features.evaluation.ablation_eval --arms semantic_only,hybrid_word_lower
```

Write the sanitized comparison to `docs/eval/ablation_summary.md`. **Report what it says, including if it contradicts the audit's "BM25 aporta ~0" nuance.** If semantic-only matches or beats hybrid on every gate with zero new misses, that is a finding worth surfacing to the owner — it is not, by itself, permission to remove the BM25 channel. If BM25 rescues even one question, the belief is refuted and the report says so.

---

### Task 2: Bilingual Snowball BM25 as a third arm

**Files:**
- Create: `src/adapters/secondary/lexical/snowball_bm25_index.py`, `requirements-experiments.txt`, `tests/test_snowball_bm25_index.py`
- Modify: `src/features/evaluation/ablation_eval.py` (`_snowball_index`), `.gitignore`
- Test: `tests/test_snowball_bm25_index.py`, `tests/test_evaluation_ablation.py`

**Interfaces:**
- Produces: `SnowballBm25Index(persist_path: Path, *, languages: tuple[str, str] = ("english", "spanish"))` implementing `LexicalIndexPort`, with `LEXICAL_PROFILE = "snowball-bilingual-v1"` and the same versioned payload shape Bucket 3 Task 2 defines.
- Depends on Bucket 3 Task 2 for `BM25_SCHEMA_VERSION` / `LEXICAL_PROFILE` / the candidate-promote split. If Bucket 3 has not landed, this task writes its own versioned payload and Bucket 3 later reconciles the two — say which you did in the commit message.

**Tokenizer:** stem every `\w+` token with both the English and the Spanish Snowball stemmer and keep both stems. Applying only one language's stemmer to a bilingual corpus is what makes a single-language index worse than none; keeping both is the cheap, honest bilingual approximation. This is a hypothesis the ablation tests — not a claim.

- [ ] **Step 1: Write the failing tests** — create `tests/test_snowball_bm25_index.py`:

```python
from __future__ import annotations

import pytest

snowballstemmer = pytest.importorskip(
    "snowballstemmer",
    reason="experiment-only dependency; install requirements-experiments.txt to run",
)

from src.adapters.secondary.lexical.snowball_bm25_index import (  # noqa: E402
    LEXICAL_PROFILE,
    SnowballBm25Index,
    bilingual_tokenize,
)


def test_profile_name_is_distinct_from_the_production_one() -> None:
    """A differently-tokenized index must be unloadable by a word-lower-v1
    runtime — the whole point of Bucket 3's lexical_profile field."""
    from src.adapters.secondary.lexical.bm25_lexical_index import LEXICAL_PROFILE as PROD

    assert LEXICAL_PROFILE == "snowball-bilingual-v1"
    assert LEXICAL_PROFILE != PROD


def test_english_and_spanish_stems_are_both_emitted() -> None:
    tokens = bilingual_tokenize("cleaning validation procedimientos")

    assert "clean" in tokens
    assert "procedimient" in tokens


def test_tokenizer_is_case_and_punctuation_insensitive() -> None:
    assert bilingual_tokenize("Cleaning, VALIDATION.") == bilingual_tokenize("cleaning validation")


def test_index_writes_its_own_profile(tmp_path) -> None:
    import json

    index = SnowballBm25Index(persist_path=tmp_path / "snowball.json")
    index.promote(index.build_index(FIXTURE_CHUNKS, chunks_sha256="aaaa"))

    assert json.loads((tmp_path / "snowball.json").read_text(encoding="utf-8"))["lexical_profile"] == LEXICAL_PROFILE
```

Import `FIXTURE_CHUNKS` from `tests/test_bm25_lexical_index.py`. Verify the exact stems in Step 3 with a one-liner before trusting the literals `"clean"` and `"procedimient"` — Snowball's real output is what the test must assert, not what looks plausible.

- [ ] **Step 2: Run the tests to verify they fail (or skip)**

Run: `pytest tests/test_snowball_bm25_index.py -q`

Expected: SKIPPED if `snowballstemmer` is absent — which is the correct default state for the repo. To work on this task, install it locally only:

```bash
pip install snowballstemmer
python -c "import snowballstemmer as s; print(s.stemmer('english').stemWord('cleaning'), s.stemmer('spanish').stemWord('procedimientos'))"
```

Use that printed output to fix the literals in the test. Then re-run: FAIL on the missing module.

- [ ] **Step 3: Implement the adapter**

```python
# src/adapters/secondary/lexical/snowball_bm25_index.py
LEXICAL_PROFILE = "snowball-bilingual-v1"

_STEMMERS = tuple(snowballstemmer.stemmer(lang) for lang in ("english", "spanish"))


def bilingual_tokenize(text: str) -> list[str]:
    """Both stems per token, deliberately. The corpus is bilingual and a chunk's
    language is not known at query time; applying a single language's stemmer
    would degrade the other half. Whether this beats word-lower-v1 at all is
    what the ablation measures — it is a hypothesis, not a claim."""
    tokens: list[str] = []
    for raw in _TOKEN_PATTERN.findall(text.lower()):
        for stemmer in _STEMMERS:
            stem = stemmer.stemWord(raw)
            if stem not in tokens[-2:]:
                tokens.append(stem)
    return tokens
```

`_STEMMERS` is a module-level *immutable* tuple of stateless objects, not a mutable cache — it does not fall under CLAUDE.md's no-module-level-mutable-singleton rule. Note that in a comment so a future reader does not "fix" it.

The rest of the class mirrors `Bm25LexicalIndex` with `bilingual_tokenize` in place of `tokenize` and its own `LEXICAL_PROFILE`. Prefer composition (hold a `Bm25LexicalIndex`-shaped builder) over inheritance if the parent's `tokenize` is not injectable; a small amount of duplication in an experiment-only adapter is cheaper than making the production class configurable for a hypothesis that may not survive.

`requirements-experiments.txt`:

```
# Experiment-only. NEVER installed into the deploy image, NEVER hashed into
# requirements-lock.txt. See docs/superpowers/plans/2026-09-04-arch-5-retrieval-experiments.md.
snowballstemmer
sentence-transformers  # already a runtime dep; listed for the reranker's CrossEncoder
```

Add `retrieval/output/experiments/` to `.gitignore`.

- [ ] **Step 4: Wire the third arm and build its index**

`_snowball_index()` in `ablation_eval.py` builds (once, into `retrieval/output/experiments/snowball_bm25.json`) and returns a loaded `SnowballBm25Index`. It must never write to `settings.bm25_path`. Add an assertion to that effect:

```python
    assert path != load_settings().bm25_path, "experiment index must not overwrite the live one"
```

- [ ] **Step 5: Run the three-arm ablation and report**

```bash
pytest tests/test_snowball_bm25_index.py tests/test_evaluation_ablation.py -q
python -m src.features.evaluation.ablation_eval
```

Update `docs/eval/ablation_summary.md` with all three arms, each judged against the four acceptance gates and the zero-new-misses rule. **Stop here and hand the report to the owner.** Task 3 is conditional on their reading it and saying to proceed — a reranker layered on an unmeasured base is exactly the kind of unvalidated addition this bucket exists to prevent.

---

### Task 3: `RerankerPort` and `bge-reranker-v2-m3`, opt-in only

**Files:**
- Create: `src/adapters/secondary/reranker/__init__.py`, `src/adapters/secondary/reranker/flag_reranker.py`, `tests/test_reranker_adapter.py`
- Modify: `src/domain/ports.py`, `src/features/retrieval/use_cases.py:24-73`, `tests/test_hybrid_retriever_use_case.py`
- Test: `tests/test_hybrid_retriever_use_case.py`, `tests/test_reranker_adapter.py`

**Interfaces:**
- Produces:
  - `RerankerPort` in `src/domain/ports.py`:
    ```python
    @runtime_checkable
    class RerankerPort(Protocol):
        def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]: ...
    ```
    `candidates` is `(chunk_id, text)`; the return is `(chunk_id, score)` best-first over **exactly the same id set**.
  - `FlagReranker(model_name: str = "BAAI/bge-reranker-v2-m3", *, batch_size: int = 16)` implementing it.
  - `HybridRetriever(vector_store, lexical_index, expansion_mode="off", *, reranker: RerankerPort | None = None, rerank_window: int = 20)`. `reranker=None` is the default and the served configuration; the code path is byte-identical to today's when it is `None`.

**The gate-preservation property — the load-bearing design constraint.** `QueryUseCase._answer_question_impl` calls `retrieve(question, k=SEMANTIC_EXTRACTION_K)` (40), reads the refusal score from whichever result has `semantic_rank == 1` anywhere in that list, and only then slices `results[:PROMPT_CONTEXT_K]` for the prompt. Therefore the reranker must:

1. permute **only** the first `rerank_window` (20) fused results,
2. leave positions 21+ in their original fused order (the tail), and
3. **never add, drop, or deduplicate** an element.

Under those three rules the semantic-rank-1 result is still somewhere in the returned list, so `top1_semantic_score_from_results` returns the same float and the refusal gate is provably unchanged — the reranker changes only *which chunks reach the prompt*, never *whether the question is answered*. Any design that truncates before the gate reads it would silently retune `0.5999` and `0.5500`, which this project forbids.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hybrid_retriever_use_case.py`:

```python
class _ReversingReranker:
    """Deterministic, model-free: reverses whatever window it is handed. Enough
    to prove the windowing/tail/gate contract without loading a cross-encoder."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def rerank(self, query: str, candidates):
        ids = [chunk_id for chunk_id, _ in candidates]
        self.seen.append(list(ids))
        return [(chunk_id, float(i)) for i, chunk_id in enumerate(reversed(ids))]


def test_reranker_permutes_only_the_window_and_keeps_the_tail() -> None:
    retriever = _retriever_with(30, reranker=_ReversingReranker(), rerank_window=20)

    results = retriever.retrieve("q", k=30)
    ids = [r.chunk_id for r in results]
    baseline = [r.chunk_id for r in _retriever_with(30).retrieve("q", k=30)]

    assert ids[:20] == list(reversed(baseline[:20]))
    assert ids[20:] == baseline[20:]
    assert sorted(ids) == sorted(baseline)


def test_reranker_never_changes_the_refusal_gate_input() -> None:
    """The whole safety argument in one assertion: the score the gate reads is
    identical with and without a reranker, because semantic_rank == 1 is still
    present and its semantic_score is untouched."""
    from src.domain.policies import top1_semantic_score_from_results

    plain = _retriever_with(30).retrieve("q", k=30)
    reranked = _retriever_with(30, reranker=_ReversingReranker(), rerank_window=20).retrieve("q", k=30)

    assert top1_semantic_score_from_results(reranked) == top1_semantic_score_from_results(plain)


def test_reranker_is_off_by_default() -> None:
    retriever = _retriever_with(30)

    assert retriever._reranker is None


def test_a_reranker_that_drops_an_id_is_rejected() -> None:
    class _Dropping:
        def rerank(self, query, candidates):
            return [(candidates[0][0], 1.0)]

    with pytest.raises(ValueError, match="same id set"):
        _retriever_with(30, reranker=_Dropping(), rerank_window=20).retrieve("q", k=30)
```

Add a `_retriever_with(n, *, reranker=None, rerank_window=20)` helper to that module built on its existing fake vector store / lexical index (line 16), producing `n` deterministic hits.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hybrid_retriever_use_case.py -q -k rerank`

Expected: FAIL — `HybridRetriever.__init__() got an unexpected keyword argument 'reranker'`.

- [ ] **Step 3: Add the port and the wiring**

`src/domain/ports.py` gains `RerankerPort` as specified above (`Sequence` from `typing`; no new third-party import, so the domain stays framework-free).

`src/features/retrieval/use_cases.py` — after `fused` is built and before `return fused[:k]`:

```python
            if self._reranker is not None:
                fused = self._apply_reranker(query_text, fused)

            return fused[:k]

    def _apply_reranker(self, query_text: str, fused: list[RetrievalResult]) -> list[RetrievalResult]:
        """Permutes ONLY the first rerank_window entries and leaves the tail in
        fused order. Nothing is added, dropped or deduplicated, so the result
        with semantic_rank == 1 is still present and the refusal gate reads the
        same score it would without a reranker (see
        src.domain.policies.top1_semantic_score_from_results). A reranker that
        truncated before the gate read it would silently retune 0.5999/0.5500."""
        window, tail = fused[: self._rerank_window], fused[self._rerank_window :]
        if len(window) < 2:
            return fused
        by_id = {result.chunk_id: result for result in window}
        scored = self._reranker.rerank(
            query_text, [(r.chunk_id, str(r.metadata.get("chunk_text", ""))) for r in window]
        )
        if {chunk_id for chunk_id, _ in scored} != set(by_id):
            raise ValueError("reranker must return the same id set it was given")
        return [by_id[chunk_id] for chunk_id, _ in scored] + tail
```

`RetrievalResult` fields (`semantic_rank`, `semantic_score`, `fused_score`) are carried through untouched — reranking reorders objects, it never rewrites their scores. That is what keeps the gate and every downstream report interpretable.

- [ ] **Step 4: Add the model-backed adapter**

```python
# src/adapters/secondary/reranker/flag_reranker.py
MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class FlagReranker:
    """Implements RerankerPort with a sentence-transformers CrossEncoder. The
    model is loaded lazily on first rerank() and held on the instance — the
    composition root owns its lifetime, exactly like the embedder. NOT wired
    into serving: adding ~2.3GB to the image and a cross-encoder pass to every
    query is a deploy decision with its own latency budget, not a default."""
```

Its tests mock `CrossEncoder` — no model download in CI, matching the SDK-boundary mocking exception.

- [ ] **Step 5: Measure it, then stop**

```bash
pytest -q
ruff check src tests
mypy src
python -m src.features.evaluation.ablation_eval --arms hybrid_word_lower,hybrid_word_lower_reranked
```

Add the reranked arm to `docs/eval/ablation_summary.md` with, in addition to the four gates and the new-misses list, a **measured p50/p95 rerank latency** over the eval set — the audit's stated unknown. Then stop.

Whether any of this ships is the owner's call and needs, separately: a passing acceptance-gate result with zero new misses, a latency budget the deploy can absorb, a lock file regenerated in a Debian Bookworm amd64 / Python 3.11 container, and an explicit authorization. Nothing in this bucket flips a default on its own.
