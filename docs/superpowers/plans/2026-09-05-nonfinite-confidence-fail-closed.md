# Non-Finite Retrieval Confidence Fail-Closed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat a non-finite (`NaN`, `+inf`, `-inf`) selected retrieval confidence as absent, so the query path takes its existing hard-refusal branch with `confidence=None` instead of reaching generation.

**Architecture:** Normalize at the one existing domain boundary that already owns "is there a usable confidence" — `RefusalPolicy.top1_semantic_score` — and apply the same finiteness requirement to `classify_score`'s missing-score guard and to the module-level `is_confident` helper so both boolean interfaces agree. `QueryUseCase` then takes its existing `hard_refuse` branch unchanged. No HTTP schema, adapter, prompt, index, or retrieval-ordering change.

**Tech Stack:** Python 3.11 (CI/prod target), stdlib `math`, pytest, existing `Settings` + `tests/fakes.py` port fakes. Ruff (line-length 120, rules `E,F,I,W`) and `mypy --strict` on `src` only. No new dependencies, no lock change.

**Spec:** `SPEC.md` (unsupported retrieval must refuse) + `docs/adr/009-phase3-grounded-review-gate.md` (band semantics). There is no separate spec document for this fix; the Problem section below is the argument, and it is derived from source read at the pinned revision, not from an observed incident.

**Source revision:** `d41d529fe562fa39425a383826090911f3ed926c` — verified as the local `HEAD` of this checkout on 2026-09-05.

---

## Problem

Read at the pinned revision, verified by line:

- `src/domain/policies.py:102-103` — `is_confident` guards `None` only: `top1_semantic_score is not None and top1_semantic_score >= threshold`. `+inf >= threshold` is `True`, so positive infinity reads as **confident**. `NaN` and `-inf` already return `False` here by comparison semantics.
- `src/domain/policies.py:150-159` — `classify_score` guards `None` only.
  - `binary`: `NaN >= threshold` is `False` → `hard_refuse` (accidentally correct); `+inf` → `confident`.
  - `grounded_review`: `NaN < review_floor` is `False` and `NaN < threshold` is `False`, so `NaN` falls through both tests and reaches **`confident`**; `+inf` → `confident`.
- `src/domain/policies.py:147-148` — `RefusalPolicy.top1_semantic_score` returns `top1_semantic_score_from_results(results)` unnormalized, so even in the branches that already refuse, the non-finite float is carried into `QueryAnswer.confidence` and into `_log_query_completed`'s structured log (where `json.dumps` emits bare `NaN`/`Infinity`, which is not valid JSON).
- `src/domain/policies.py:133-135` — the constructor already requires `threshold` and `review_floor` to be finite. The retrieved score is held to a weaker standard than the configured constants it is compared against. This fix closes that asymmetry.
- `src/features/query/use_cases.py:143-155` — the existing `hard_refuse` branch this fix routes into. It returns the canonical refusal before any prompt is built.

`tests/test_domain_policies.py:252-254` covers non-finite **constructor** arguments. Nothing in the suite covers a non-finite **retrieved** score.

**[UNCERTAIN]** The live index has not been observed emitting a non-finite score. This is input-boundary hardening justified by a deterministic source-level path, not an incident claim.

## Global Constraints

Copied from `CLAUDE.md` and ADR-009; every task's requirements implicitly include this section.

1. Byte-stable constants unchanged: refusal threshold `0.5999`, review floor `0.5500`, RRF `k=60` with ascending-`chunk_id` tie-break. No new threshold, no score clamp, no recalibration against any holdout.
2. Served defaults unchanged: `REFUSAL_POLICY=binary`, `expansion_mode="off"`, `INDEX_PROFILE=contextual-v1`, `reranker=None`.
3. Negative *finite* cosine values keep their existing behavior. Only non-finite values change.
4. No frozen dataset or report artifact is touched. Never run an integrity `--write`. No index rebuild.
5. Citation re-derivation, whole-set citation failure, verbatim evidence policy, prompts, response fields, and no-content logging are unchanged. Log no prompts, answers, keys, or new score payloads.
6. Normalize only the *selected* policy confidence. Do not mutate `RetrievalResult` objects and do not change `top1_semantic_score_from_results`'s selection algorithm.
7. Exactly three paths may appear in the final diff: `src/domain/policies.py`, `tests/test_nonfinite_confidence.py`, and this plan file. Existing tests are read-only.
8. No paid generation eval, no verdict import, no default flip, no ADR acceptance, no Space operation, no push, no merge, no PR. Commit is **local only**, and only after the owner approves the final diff.
9. Stop after three failed repair attempts and return to Explore. Do not attempt a blind fourth fix.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/domain/policies.py` | Refusal band policy | Modify 3 functions (`is_confident`, `RefusalPolicy.top1_semantic_score`, `RefusalPolicy.classify_score`) |
| `tests/test_nonfinite_confidence.py` | Judge for the new behavior + regression lock on finite/`None` semantics | Create |
| `src/features/query/use_cases.py` | Query orchestration | **Unchanged** — it already branches on `hard_refuse` |

One task: the deliverable is a single behavior change with one test cycle, and no reviewer could accept half of it.

## Execution Preflight — prerequisite, not part of the task

Run from the repo root **before** editing anything:

```bash
git rev-parse HEAD
git status --short
python -m pytest -q
python -m ruff check src tests
python -m mypy src
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.regression_set_integrity --verify
python -m src.features.evaluation.gate_holdout_integrity --verify
```

Two environment facts measured on this machine on 2026-09-05:

- In Git Bash, `pytest`/`ruff`/`mypy` are **not** on `PATH`; they must be invoked as `python -m <tool>`. Plain `pytest` exits 127.
- The local interpreter is **Python 3.14**, while the CI/prod target is 3.11. This fix uses only `math.isfinite`, whose semantics are identical across both, so the gap does not affect this task — but do not generalize a local green run into a claim about the 3.11 CI job.

Baseline actually observed at `d41d529` before this plan was written: **650 passed, 1 warning in 109.41s**, `PYTEST_EXIT=0`; Ruff `All checks passed!`, `RUFF_EXIT=0`; Mypy `Success: no issues found in 77 source files`, `MYPY_EXIT=0`; all three integrity commands `hash OK` / `hash + composition OK`, exit 0. Re-run it yourself rather than trusting this line — it is a record, not a substitute.

Required outcomes: `HEAD` is the pinned revision; full suite exits 0 with a positive passed count and zero failures; Ruff and Mypy exit 0; all three integrity commands exit 0. Report the **actual** counts — never a remembered number. Real-index tests must actually execute rather than disappearing behind missing-index skips; compare the skip list against this preflight in the final gate. If any prerequisite is red, stop and report it; do not repair unrelated baseline failures inside this task.

The suite is slow (it loads real models and queries the live index). Start it in the background, redirect to a log **outside** the repo, capture its exit code, and poll; never advance to a dependent step while a check is still running.

---

## Task 1: Fail closed on non-finite retrieval confidence

**Files:**
- Create: `tests/test_nonfinite_confidence.py`
- Modify: `src/domain/policies.py:102-103`, `src/domain/policies.py:147-148`, `src/domain/policies.py:150-159`
- Test: `tests/test_nonfinite_confidence.py`
- Read-only regression coverage: `tests/test_domain_policies.py`, `tests/test_core_config.py`, `tests/test_query_use_case.py`, `tests/test_adversarial_challenger.py`, `tests/test_import_invariants.py`, plus the full suite.

**Interfaces:**

- **Consumes** (all unchanged signatures, verified at the pinned revision):

```python
is_confident(top1_semantic_score: Optional[float], threshold: float) -> bool
RefusalPolicy.__init__(self, threshold: float, *, mode: RefusalPolicyName = "binary", review_floor: float = 0.5500) -> None
RefusalPolicy.top1_semantic_score(self, results: Sequence[RetrievalResult]) -> Optional[float]
RefusalPolicy.classify_score(self, score: Optional[float]) -> GateBand
RefusalPolicy.classify(self, results: Sequence[RetrievalResult]) -> GateBand
QueryUseCase.__init__(self, retriever: RetrieverPort, llm_client: LLMClientPort, settings: Settings) -> None
QueryUseCase.answer_question(self, question: str, language: Language) -> QueryAnswer   # async
InMemoryRetriever.__init__(self, results: list[RetrievalResult]) -> None
InMemoryLLMClient.__init__(self, response=None, error: Optional[Exception] = None) -> None
```

- **Produces:** no new public names or types. Changed semantics only:
  - `is_confident(non_finite, t)` → `False` (was `True` for `+inf`).
  - `RefusalPolicy.classify_score(non_finite)` → `"hard_refuse"` in both modes (was `"confident"` for `+inf` in both, and for `NaN` in `grounded_review`).
  - `RefusalPolicy.top1_semantic_score(...)` → `None` when the selected score is non-finite.
  - Resulting `QueryAnswer`: `refused=True`, `status="ok"`, `gate_band="hard_refuse"`, `confidence=None`, `answer == REFUSAL_MESSAGE[language]`, `citations == []`.
  - All finite and `None` behavior is unchanged.

### Steps

- [ ] **Step 1: Confirm the test path is free**

```bash
test ! -e tests/test_nonfinite_confidence.py; echo "EXIT=$?"
```

Expected: `EXIT=0`. If the file already exists, stop and report — do not overwrite.

- [ ] **Step 2: Write the failing test**

Create `tests/test_nonfinite_confidence.py` with exactly this content. Do not touch production code in this step.

```python
from __future__ import annotations

import asyncio

import pytest

from src.core.config import RefusalPolicyName, Settings
from src.domain.models import GateBand, Language, RetrievalResult
from src.domain.policies import RefusalPolicy, is_confident
from src.features.query.prompts import REFUSAL_MESSAGE
from src.features.query.use_cases import QueryUseCase
from tests.fakes import InMemoryLLMClient, InMemoryRetriever

THRESHOLD = 0.5999
FLOOR = 0.5500
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def _result(score: float | None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="fixture::chunk-0000",
        fused_score=1.0 / 61,
        semantic_rank=1,
        semantic_score=score,
        bm25_rank=None,
        bm25_score=None,
        metadata={
            "document_id": "fixture",
            "document_title": "Fixture",
            "section_heading": "Procedure",
            "revision": "Rev A",
            "source_type": "synthetic",
            "chunk_text": "This complete fixture context must not reach generation.",
        },
    )


@pytest.mark.parametrize("mode", ["binary", "grounded_review"])
@pytest.mark.parametrize("score", NON_FINITE)
def test_nonfinite_policy_confidence_fails_closed(mode: RefusalPolicyName, score: float) -> None:
    policy = RefusalPolicy(THRESHOLD, mode=mode, review_floor=FLOOR)
    results = [_result(score)]

    assert policy.classify_score(score) == "hard_refuse"
    assert policy.classify(results) == "hard_refuse"
    assert policy.top1_semantic_score(results) is None


@pytest.mark.parametrize("score", NON_FINITE)
def test_nonfinite_scalar_confidence_is_false(score: float) -> None:
    assert is_confident(score, THRESHOLD) is False


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("mode", ["binary", "grounded_review"])
@pytest.mark.parametrize("score", NON_FINITE)
def test_nonfinite_query_refuses_without_generation(
    language: Language, mode: RefusalPolicyName, score: float
) -> None:
    settings = Settings(
        groq_api_key="groq-test-key",
        openai_api_key="openai-test-key",
        llm_provider="groq",
        refusal_cosine_threshold=THRESHOLD,
        refusal_policy=mode,
        refusal_review_floor=FLOOR,
        log_level="INFO",
    )
    llm = InMemoryLLMClient(error=AssertionError("generation must not be called"))
    use_case = QueryUseCase(InMemoryRetriever([_result(score)]), llm, settings)

    answer = asyncio.run(use_case.answer_question("Fixture question", language))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.gate_band == "hard_refuse"
    assert answer.confidence is None
    assert answer.answer == REFUSAL_MESSAGE[language]
    assert answer.citations == []
    assert answer.threshold == THRESHOLD
    assert answer.review_floor == (FLOOR if mode == "grounded_review" else None)


@pytest.mark.parametrize(
    "score,binary_band,review_band",
    [
        (None, "hard_refuse", "hard_refuse"),
        (-0.1, "hard_refuse", "hard_refuse"),
        (0.5500, "hard_refuse", "grounded_review"),
        (0.5998, "hard_refuse", "grounded_review"),
        (0.5999, "confident", "confident"),
        (0.7, "confident", "confident"),
    ],
)
def test_finite_and_missing_confidence_keep_existing_behavior(
    score: float | None, binary_band: GateBand, review_band: GateBand
) -> None:
    result = _result(score)
    binary = RefusalPolicy(THRESHOLD)
    review = RefusalPolicy(THRESHOLD, mode="grounded_review", review_floor=FLOOR)

    assert binary.classify([result]) == binary_band
    assert review.classify([result]) == review_band
    assert binary.top1_semantic_score([result]) == score
    assert review.top1_semantic_score([result]) == score
    assert is_confident(score, THRESHOLD) == (binary_band == "confident")
    assert result.semantic_score == score
```

Notes for the implementer:
- `fixture::chunk-0000` is a unit-test fixture id, deliberately not a corpus or holdout anchor.
- Port fakes are the established convention for `QueryUseCase` tests (`tests/test_query_use_case.py`); this claims no live-retrieval coverage.
- The fixture metadata is complete on purpose, so a `+inf` score reaches generation in RED rather than being diverted by the `prompt_metadata_incomplete` guard.
- Import order (`src.*` then `tests.*`, alphabetical) matches `tests/test_query_use_case.py` and satisfies Ruff's `I` rules.
- The last test is a **regression lock**: it must pass both before and after the implementation.

- [ ] **Step 3: Run the test to verify it fails**

```bash
python -m pytest tests/test_nonfinite_confidence.py -q
```

Expected: **19 failed, 8 passed**, exit 1. The predicted breakdown, derived from the current comparisons:

| Case group | Current behavior | Fails |
|---|---|---|
| `classify_score` × 6 (2 modes × 3 values) | `binary/NaN`, `binary/-inf`, `review/-inf` already `hard_refuse`; `binary/+inf`, `review/NaN`, `review/+inf` are `confident` | 6 — the three already-refusing cases still fail the `top1_semantic_score(...) is None` assertion |
| scalar `is_confident` × 3 | `NaN`→`False`, `-inf`→`False`, `+inf`→`True` | 1 (`+inf`) |
| query path × 12 (2 langs × 2 modes × 3 values) | refusing cases carry the non-finite float as `confidence`; `confident` cases raise `AssertionError` out of the un-caught fake (only `GenerationError` is caught, at `use_cases.py:211`) | 12 |
| finite/`None` regression lock × 6 | already correct | 0 |

A collection error, an `ImportError`, or a missing tool is **not** RED. If the observed breakdown differs from this table, investigate before implementing — never edit an assertion to manufacture RED. This one intentional multi-case RED run is not a repair attempt and must not be counted against the three-attempt stop.

- [ ] **Step 4: Freeze the judge**

```bash
sha256sum tests/test_nonfinite_confidence.py | tee "$TEMP/nonfinite-test-red.sha256"
```

The test file is read-only from here on. If an expectation turns out to be wrong, stop, explain, and return to Step 2 — do not adapt the judge to the implementation.

- [ ] **Step 5: Write the minimal implementation**

Three replacements in `src/domain/policies.py`. `math` is already imported (the constructor uses `math.isfinite`); add no imports, reformat nothing else, and preserve the file's existing line endings.

Replace `is_confident` (currently lines 102-103):

```python
def is_confident(top1_semantic_score: Optional[float], threshold: float) -> bool:
    return (
        top1_semantic_score is not None
        and math.isfinite(top1_semantic_score)
        and top1_semantic_score >= threshold
    )
```

Replace `RefusalPolicy.top1_semantic_score` (currently lines 147-148):

```python
    def top1_semantic_score(self, results: Sequence[RetrievalResult]) -> Optional[float]:
        """A non-finite selected score is treated as absent, not as evidence:
        NaN passes neither band comparison and +inf passes both, so without
        this the gate would read a malformed score as maximum confidence."""
        score = top1_semantic_score_from_results(results)
        return score if score is not None and math.isfinite(score) else None
```

Replace `RefusalPolicy.classify_score`'s guard (currently line 151):

```python
        if score is None or not math.isfinite(score):
            return "hard_refuse"
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
python -m pytest tests/test_nonfinite_confidence.py -q
sha256sum -c "$TEMP/nonfinite-test-red.sha256"
git diff -- src/domain/policies.py
```

Expected: **27 passed**, exit 0; `tests/test_nonfinite_confidence.py: OK`; the diff shows only the three hunks above. The implementer cannot certify this alone — Step 8's reviewer reads the real output and exit codes.

- [ ] **Step 7: Run the full regression and integrity gate**

Start the suite in the background against a log outside the repo, then check each exit code.

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.regression_set_integrity --verify
python -m src.features.evaluation.gate_holdout_integrity --verify
git diff --check
git status --short
```

Expected: suite exit 0 with a positive passed count and zero failures; Ruff `All checks passed!`; Mypy `Success: no issues found ...`; three integrity commands exit 0 with their real success messages. Compare the skipped-test list against the preflight — real-index tests must still run.

**`git diff --check` will NOT be empty, and that is correct here.** `src/domain/policies.py` is committed with **CRLF** line endings at `d41d529` (328 CRLF lines), so `--check` reports "trailing whitespace" for the terminating `CR` on every *added* line — it inspects added lines only, which is why unchanged context is silent. Do not convert the file to LF to silence it: that would turn a 14-line diff into a whole-file rewrite and violate the minimal-diff constraint. Verify instead that the warnings are confined to added lines in that one file, and that no added line has a genuine trailing space or tab. Do not fabricate totals or elapsed times, and do not claim any new Recall/MRR result: no retrieval eval or paid generation run is needed to prove a local numeric guard.

`git diff --stat` omits untracked files, so stage-intent the new paths for review without committing:

```bash
git add -N -- tests/test_nonfinite_confidence.py docs/superpowers/plans/2026-09-05-nonfinite-confidence-fail-closed.md
git diff --stat
git diff --name-only
```

Expected: exactly the three authorized paths, nothing else.

- [ ] **Step 8: Independent review**

Give a **fresh** reviewer only the full diff plus these acceptance criteria — spec compliance first, then a separate fresh context for quality/minimality:

1. All three non-finite values fail closed in both `binary` and `grounded_review`.
2. The query path returns the canonical EN and ES refusal with `confidence=None` and zero LLM calls.
3. Finite and `None` semantics are byte-identical to the baseline.
4. The test file's SHA-256 is unchanged since Step 4.
5. Frozen constants, datasets, and report artifacts are untouched.
6. Exactly three paths changed.

The reviewer inspects real command output and exit codes, not the implementer's summary. Stop after three failed repair attempts. Then present the diff, the test output, and the rulings ledger to the owner and **wait**.

- [ ] **Step 9: Local commit — only after the owner approves the final diff**

```bash
git add -- src/domain/policies.py tests/test_nonfinite_confidence.py docs/superpowers/plans/2026-09-05-nonfinite-confidence-fail-closed.md
git diff --cached --check
git diff --cached --stat
git commit -m "fix(domain): fail closed on non-finite retrieval confidence"
git log --oneline -3
git status --short
```

Expected: one local Conventional Commit containing only the approved paths. Report the actual hash from the output — never invent one. Never push, never merge. Plan approval is not final-diff approval.

---

## Rulings Ledger

- **Non-finite is treated as absent, not clamped.** This is an explicit behavior change, and it is the conservative extension of the existing missing-confidence path. Cost if wrong: the system abstains on a malformed retrieval result instead of attempting generation. No finite threshold is recalibrated.
- **A non-finite rank-1 score refuses even when a lower-ranked result has a good finite score.** `top1_semantic_score_from_results`'s `max(scored)` fallback only runs when *no* result has `semantic_rank == 1`, and normalizing after it is deliberate. Redefining mixed-score-list handling is a separate decision, out of scope here.
- **No new `DecisionReason`.** A non-finite score reports the existing `below_binary_threshold` / `below_review_floor`, which is imprecise but keeps the HTTP response schema untouched. Adding a reason is a separate diff.
- **`is_confident`'s module-level form is only called from tests** (`RefusalPolicy.is_confident` is what `regression_eval.py:27` and `retrieval_eval.py:34` use, and it routes through `classify`). It is hardened anyway so the two boolean interfaces cannot disagree.
- **Structured-log side benefit, not a goal.** `_log_query_completed` currently receives the raw score; `json.dumps(float("nan"))` emits bare `NaN`, which is invalid JSON. After this fix the value is `None`. Do not add logging changes on top of that.
- **Pre-existing CRLF on `policies.py` is left alone, and it is a real repo defect worth its own task.** The file was LF through `69de33d` and is CRLF at `d41d529`; `7dd145d` ("bucket 3 — index ownership...") converted it, and that already-merged commit itself trips `git diff --check` with 656 warning lines. `src/features/query/use_cases.py`, `tests/test_domain_policies.py` and `tests/fakes.py` are CRLF too, while `src/domain/models.py` and `src/core/config.py` are LF — the repo is mixed, and `.gitattributes` only pins `start.sh`/`Dockerfile`/`nginx.conf`. Normalizing is a separate, whole-file diff and must not ride along with a behavior fix.
- **Reranker code stays unchanged.** `src/features/retrieval/use_cases.py:95-105` already checks result count and id-set equality, so a duplicate-rejection "fix" would have no honest RED.
- **No retrieval retuning.** The ablation, fusion, and floor sweeps are already measured and merged; they authorize neither a default flip nor reuse of a measurement set as a calibration set.
- **ADR-009 stays Proposed.** Its line 220 still says the import rewrites `comparison.md` while `SPEC.md`/`CLAUDE.md` and the implementation use `comparison.import.md`. SPEC wins; fixing that sentence is a separate diff and must never be followed literally.

## Owner-Pending (outside this plan)

- Approve this plan, then approve the final diff before the local commit.
- Everything gated on the paid `gate_generation_eval` run, blind grading, verdict import, the `REFUSAL_POLICY` default flip, ADR-009 acceptance, and any Space deployment stays owner-only and untouched here.
