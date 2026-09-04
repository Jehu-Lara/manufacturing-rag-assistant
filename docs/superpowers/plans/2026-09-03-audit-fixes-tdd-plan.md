# Audit Fixes (P1 Verdict) TDD Implementation Plan — v4 FINAL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix audit findings #1–#5 behind failing-first tests, incorporating the second P1 verdict's FAIL corrections (v3 deltas: exact repeat matrix from sealed expected sets, byte-immutable manifest with separate import artifacts, single-event-loop offline runners, provider-keyed fingerprinted client cache with lock, try/finally placed before state assignment), without moving any byte-stable invariant.

**Architecture:** Five tasks, each RED→GREEN with existing fakes/mocks. T1 enforces exact repeat set {1,2,3} plus exactly-one-row per (qid, policy) cell against sealed expected sets, and makes verdict import single-use without touching sealed files. T2 pins both fallback branches under fail-fast. T5 caches SDK clients by provider (fingerprint, never the secret; lock-guarded), closes idempotently, runs each offline runner inside one owning `asyncio.run`, and places lifespan teardown before state assignment. T5 builds on T2's constructor shape; all others are order-free. No new modules.

**Tech Stack:** Python 3.11, pytest, unittest.mock (AsyncMock/MagicMock), FastAPI TestClient, groq 1.7.0 / openai 3.x SDKs (both expose coroutine `close()` — verified against the installed venv; `asyncio.Lock()` takes no loop argument on 3.10+, so it is safe to construct in `__init__`).

**Spec:** `SPEC.md` + owner's P1 verdicts (chat 2026-09-03, v1 + v2 corrections + FAIL review). Execution (code changes, commits) is NOT authorized — this plan only. 0 commits mandatory.

## Global Constraints

- No commits without the owner's explicit request. Every task ends at green tests, never at `git commit`.
- `REFUSAL_COSINE_THRESHOLD` stays `0.5999`, `REFUSAL_REVIEW_FLOOR` stays `0.5500`, RRF `k=60` with ascending-`chunk_id` tie-break, `REFUSAL_POLICY` default stays `binary`, `expansion_mode` default stays `off`, served profile stays `contextual-v1`.
- Frozen datasets (`eval/eval_set.json`, `eval/regression_queries.json`, `eval/gate_holdout_v1.0.0.json`) are never re-stamped; `eval/reports/*_v1.*` baselines are never overwritten.
- After each task: `pytest <touched-test-file> -q` green; at the end: full `pytest -q`, `ruff check src tests`, `mypy src` green.
- Test commands run from the repo root with the project venv (CI uses bare `pytest`; locally `.\.venv\Scripts\python.exe -m pytest`).

---

## File Structure

- `src/features/evaluation/gate_generation_eval.py` — Task 1: `evaluate_gates` repeats block (lines 659–676) enforces exact set + matrix against caller-supplied sealed sets; `run()` gate call sites (lines 968, 1036) pass sealed sets; `import_verdicts_into_run` (lines 986–1055) gains single-use guard, seals `run_manifest.json`, writes `import_manifest.json` + `comparison.import.md`, never rewrites sealed files. Task 5-offline: `run_matrix` (lines 316–338) runs inside one owning `asyncio.run` with per-repeat close.
- `src/features/evaluation/generation_eval.py` — Task 5-offline: `_build_row` (line 116, no test callers — verified by grep) becomes `async _build_row_async`; `_build_use_case_and_retriever` (lines 48–55, no test callers — verified by grep) returns the owned client; `run()` (lines 259–264) drives one owning loop and closes the owned client inside it.
- `src/adapters/secondary/llm/groq_openai_client.py` — Task 2: `__init__` (lines 252–260) gains `rate_limit_backoff_seconds`; `_get_provider_response` (lines 447–488) uses it. Task 5: same file gains `hashlib` import, provider-keyed fingerprinted cache, `asyncio.Lock`, `_sdk_client`, idempotent `aclose()`.
- `src/adapters/primary/http/app.py` — Task 2: lifespan line 54 passes fail-fast backoff. Task 3: line 96 gains the header. Task 5: `try` opens immediately after line 54, wrapping state assignment (lines 56–59), `RateLimiter`, and `yield`, with `finally: await llm_client.aclose()`.
- `src/features/query/use_cases.py` — Task 4: metadata guard after the hard-refuse block (line 160), before prompt build (line 162).
- `src/domain/models.py` — Task 4: `DecisionReason` union (lines 17–32) gains `"incomplete_retrieved_metadata"`.
- Tests: `tests/test_evaluation_gate_generation_eval.py` (Tasks 1, 5-offline), `tests/test_llm_adapter.py` (Tasks 2, 5), `tests/test_http_app_startup.py` (Tasks 2b, 3, 5-lifespan), `tests/test_query_use_case.py` (Task 4), `tests/test_evaluation_generation_eval.py` (Task 5-offline).

---

### Task 1: repeats_conform enforces sealed expected sets + exact matrix; byte-immutable manifest with single-use import

**Files:**
- Modify: `src/features/evaluation/gate_generation_eval.py:533-560` (`evaluate_gates` signature), `:659-676` (repeats block), `:968` and `:1036` (gate call sites), `:986-1055` (`import_verdicts_into_run`)
- Test: `tests/test_evaluation_gate_generation_eval.py` (reuse local `_outcome` at line 207; update `test_repeats_conform_gate_requires_full_and_canary_repeats` at line 551; update flow-test assertions at lines 545–548; extend `test_run_end_to_end_with_injected_fakes` at line 493)

**Interfaces:**
- Consumes: `gge.FULL_REPEATS`, `gge.CANARY_REPEATS`, `gge._POLICIES`, `gge.CANARY_MUST_ANSWER`, `gge.CANARY_MUST_REFUSE`, sealed `RetrievalSnapshot.question_id` rows.
- Produces: `evaluate_gates(holdout, canary, verdicts=None, expected_holdout_ids: frozenset[str] | None = None, expected_canary_ids: frozenset[str] | None = None)` (`manifest` parameter removed as dead — repeats never consult it; both production call sites updated, inflated-manifest test deleted as inexpressible). `None` = observed-only seam for unit tests; both production call sites always pass sealed sets. `repeats_conform` PASS requires: observed question set == sealed expected set (absent or extra rows fail) AND, per (qid, policy) cell, the repeat list equals exactly `[1..N]` (duplicates or gaps fail list equality).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_evaluation_gate_generation_eval.py`:

```python
# NOTE: no inflated-manifest test exists by design — evaluate_gates takes no
# manifest input at all (removed as dead after repeats moved to sealed
# expected sets), so the falsification scenario is inexpressible. The
# wrong-set/matrix/absent/duplicate tests below pin the behavior instead.


def test_repeats_conform_rejects_wrong_repeat_set():
    holdout = [
        _outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (2, 3, 4)
    ]
    canary = [
        _outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (1, 2, 3)
    ]
    gates = {g.name: g for g in gge.evaluate_gates(holdout, canary)}
    assert gates["repeats_conform"].passed is False
    assert "[2, 3, 4]" in gates["repeats_conform"].detail


def test_repeats_conform_rejects_incomplete_policy_matrix():
    holdout = [_outcome(policy="grounded_review", repeat=r) for r in (1, 2, 3)]
    canary = [
        _outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (1, 2, 3)
    ]
    gates = {g.name: g for g in gge.evaluate_gates(holdout, canary)}
    assert gates["repeats_conform"].passed is False


def test_repeats_conform_rejects_absent_question():
    holdout = [
        _outcome(policy=p, repeat=r, question_id="h1")
        for p in ("binary", "grounded_review")
        for r in (1, 2, 3)
    ]
    canary = [
        _outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (1, 2, 3)
    ]
    gates = {
        g.name: g
        for g in gge.evaluate_gates(
            holdout,
            canary,
            expected_holdout_ids=frozenset({"h1", "h2"}),
            expected_canary_ids=frozenset({"x"}),
        )
    }
    assert gates["repeats_conform"].passed is False
    assert "missing ['h2']" in gates["repeats_conform"].detail


def test_repeats_conform_rejects_duplicate_rows():
    holdout = [
        _outcome(policy=p, repeat=r, question_id="h1")
        for p in ("binary", "grounded_review")
        for r in (1, 2, 3)
    ] + [_outcome(policy="binary", repeat=1, question_id="h1")]
    canary = [
        _outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (1, 2, 3)
    ]
    gates = {
        g.name: g
        for g in gge.evaluate_gates(
            holdout,
            canary,
            expected_holdout_ids=frozenset({"h1"}),
            expected_canary_ids=frozenset({"x"}),
        )
    }
    assert gates["repeats_conform"].passed is False


def test_repeats_conform_rejects_fully_absent_canary():
    present = ("r001", "r002", "r018", "r019")
    canary = [
        _outcome(policy=p, repeat=r, question_id=qid)
        for qid in present
        for p in ("binary", "grounded_review")
        for r in (1, 2, 3)
    ]
    holdout = [
        _outcome(policy=p, repeat=r, question_id="h1")
        for p in ("binary", "grounded_review")
        for r in (1, 2, 3)
    ]
    gates = {
        g.name: g
        for g in gge.evaluate_gates(
            holdout,
            canary,
            expected_holdout_ids=frozenset({"h1"}),
            expected_canary_ids=frozenset({"r001", "r002", "r018", "r019", "r020"}),
        )
    }
    assert gates["repeats_conform"].passed is False
    assert "missing ['r020']" in gates["repeats_conform"].detail
```

(`_outcome` defaults: `policy="grounded_review"`, `repeat=1`, `question_id="x"` — the canary fixtures reuse the `"x"` default, matching `expected_canary_ids={"x"}`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_gate_generation_eval.py -q -k "repeats_conform"`
Expected: wrong-set FAILS (bare count of 3 passes); incomplete-matrix FAILS (single-policy data passes the count); absent-question, duplicate-rows and fully-absent-canary FAIL (observed-only grouping cannot see them). (The original inflated-manifest reproduction is retired: with no manifest input, the falsification is structurally impossible.)

- [ ] **Step 3: Write minimal implementation** — extend the signature:

```python
def evaluate_gates(
    holdout: list[QuestionOutcome],
    canary: list[QuestionOutcome],
    verdicts: Optional[HumanVerdicts] = None,
    manifest: Optional[dict[str, Any]] = None,
    expected_holdout_ids: frozenset[str] | None = None,
    expected_canary_ids: frozenset[str] | None = None,
) -> list[GateResult]:
```

(`frozenset[str] | None` needs no import — the file has `from __future__ import annotations`.) Replace the manifest-precedence block (lines 659–676) with:

```python
    repeat_problems: list[str] = []
    for label, rows, repeats, sealed in (
        ("holdout", holdout, FULL_REPEATS, expected_holdout_ids),
        ("canary", canary, CANARY_REPEATS, expected_canary_ids),
    ):
        want = list(range(1, repeats + 1))
        cells: dict[str, list[QuestionOutcome]] = {}
        for row in rows:
            cells.setdefault(row.question_id, []).append(row)
        if sealed is not None:
            missing = sorted(set(sealed) - set(cells))
            extra = sorted(set(cells) - set(sealed))
            if missing or extra:
                repeat_problems.append(f"{label}: question set mismatch - missing {missing}, extra {extra}")
        if not cells:
            repeat_problems.append(f"{label}: no outcomes")
            continue
        for qid in sorted(cells):
            for policy in _POLICIES:
                got = sorted(r.repeat for r in cells[qid] if r.policy == policy)
                if got != want:
                    repeat_problems.append(f"{label}/{qid}/{policy}: repeats {got} != {want}")
    gates.append(
        GateResult(
            "repeats_conform",
            not repeat_problems,
            "; ".join(repeat_problems)
            if repeat_problems
            else f"holdout x{FULL_REPEATS} (needs {FULL_REPEATS}), canary x{CANARY_REPEATS} (needs {CANARY_REPEATS})",
        )
    )
```

Then update the existing `test_repeats_conform_gate_requires_full_and_canary_repeats` full case (lines 558–559) so the passing fixture actually holds a complete matrix:

```python
    canary_full = [_outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (1, 2, 3)]
    holdout_full = [_outcome(policy=p, repeat=r) for p in ("binary", "grounded_review") for r in (1, 2, 3)]
```

(the single `_outcome` default `question_id="x"` puts the whole matrix in one cell — exactly what the gate inspects; the `"needs 3" in detail` assertion on the passing path is preserved verbatim by the else-branch). The single-repeat assertions in that test keep failing as before.

- [ ] **Step 4: Pass sealed sets at both production call sites.** In `run()` (line 968):

```python
    gates = evaluate_gates(
        holdout_outcomes,
        canary_outcomes,
        expected_holdout_ids=frozenset(q["id"] for q in holdout_questions),
        expected_canary_ids=frozenset(q["id"] for q in canary_questions),
    )
```

(`holdout_questions` comes from the integrity-verified holdout file; `canary_questions` from the integrity-verified regression set — the sealed datasets.) In `import_verdicts_into_run` (line 1036), derive the sets from the sealed `retrieval.jsonl` snapshots (already loaded there at lines 1023–1026; canary ids are the module constants):

```python
    canary_ids = set((*CANARY_MUST_ANSWER, *CANARY_MUST_REFUSE))
    gates = evaluate_gates(
        holdout,
        canary,
        verdicts,
        expected_holdout_ids=frozenset(s.question_id for s in snapshots if s.question_id not in canary_ids),
        expected_canary_ids=frozenset(canary_ids),
    )
```

`expected_canary_ids` comes from the module constants directly, never from snapshots — a canary absent from both outcomes and snapshots is still detected as missing. (Known limitation, documented: the holdout side has no constant list, so import derives it from sealed snapshots; a holdout question absent from both files would pass import-time. The `run()` call site is airtight — it passes the integrity-verified holdout file's ids.)

- [ ] **Step 5: Byte-immutable manifest with separate import artifacts.** In `import_verdicts_into_run`: insert as the very first lines (before the checksums parsing):

```python
    if (run_dir / "checksums.import.txt").exists():
        raise ValueError(
            f"{run_dir} already has checksums.import.txt - verdict imports are single-use; "
            "re-running would overwrite the import artifacts and re-seal them"
        )
    checksums_file = run_dir / "checksums.txt"
    if not checksums_file.exists():
        raise ValueError(
            f"{run_dir} has no checksums.txt - refusing to import verdicts "
            "without the sealed hashes to verify against"
        )
```

Add `"run_manifest.json"` to the immutable tuple (lines 1000–1005):

```python
        immutable_files = (
            "run_manifest.json",
            "outcomes.jsonl",
            "retrieval.jsonl",
            "blind_checklist.baseline.json",
            "arm_map.sealed.json",
        )
```

(`_finalize_dir` already seals `run_manifest.json` inside `checksums.txt` at write time, and nothing legitimately mutates it between write and import.) Replace the manifest rewrite (lines 1037–1040) and `comparison.md` rewrite (lines 1044–1046) with separate artifacts — the sealed files are never touched:

```python
    gates = evaluate_gates(holdout, canary, verdicts, ...)
    render_manifest = {**manifest, "verdicts_imported": True}
    (run_dir / "import_manifest.json").write_text(
        json.dumps({"verdicts_imported": True, **asdict(verdicts)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "comparison.import.md").write_text(
        render_comparison(render_manifest, snapshots, holdout, canary, gates), encoding="utf-8"
    )
    imported_checksums = [
        f"{_sha256_file(p)}  {p.name}"
        for p in sorted(run_dir.iterdir())
        if p.name not in ("checksums.txt", "checksums.import.txt")
    ]
    (run_dir / "checksums.import.txt").write_text("\n".join(imported_checksums) + "\n", encoding="utf-8")
```

(`asdict` is already imported at line 10; `human_verdicts.json` is folded into `import_manifest.json` — no existing test asserts that filename.) `checksums.import.txt` is written last, so its existence is the single-use completion marker. Import writes are sequential full-file overwrites with the marker last: a crash mid-import leaves partial artifacts without a marker and allows a clean retry (every write regenerates fully), but this is retry-safe, not a collective atomic transaction — documented limitation. The whole verification is fail-closed: missing `checksums.txt`, malformed lines (`<sha256>  <filename>` shape enforced per line), a sealed name absent from it, or a missing immutable file each raise `ValueError` instead of skipping.

- [ ] **Step 6: Update the existing flow-test assertions and add the single-use test.** In `test_run_end_to_end_with_injected_fakes`, replace lines 546–548:

```python
    assert json.loads((run_dir / "run_manifest.json").read_text())["verdicts_imported"] is False
    assert "citation=" in (run_dir / "comparison.import.md").read_text()
    assert json.loads((run_dir / "import_manifest.json").read_text())["verdicts_imported"] is True
    assert (run_dir / "checksums.import.txt").exists()

    with pytest.raises(ValueError, match="single-use"):
        gge.import_verdicts_into_run(run_dir)
```

Run the appended `pytest.raises` block before the Step-5 edit to watch it fail (second import silently succeeds), then after. Add the manifest tamper test (mirrors `test_import_verdicts_rejects_tampered_immutable_file` at line 564):

```python
def test_import_verdicts_rejects_tampered_manifest(tmp_path: Path):    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outcomes.jsonl").write_text("orig\n", encoding="utf-8")
    (run_dir / "retrieval.jsonl").write_text("orig\n", encoding="utf-8")
    (run_dir / "blind_checklist.baseline.json").write_text("{}", encoding="utf-8")
    (run_dir / "arm_map.sealed.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text("orig\n", encoding="utf-8")
    (run_dir / "blind_checklist.csv").write_text("a,b\n", encoding="utf-8")
    checksums = "\n".join(f"{gge._sha256_file(p)}  {p.name}" for p in sorted(run_dir.iterdir()))
    (run_dir / "checksums.txt").write_text(checksums + "\n", encoding="utf-8")

    (run_dir / "run_manifest.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tampered"):
        gge.import_verdicts_into_run(run_dir)
```


```python
def test_import_verdicts_rejects_missing_checksums(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="no checksums.txt"):
        gge.import_verdicts_into_run(run_dir)
```

- [ ] **Step 7: Run the file's tests to verify all pass**

Run: `pytest tests/test_evaluation_gate_generation_eval.py -q`
Expected: PASS — other `evaluate_gates` callers in that file (lines 247, 265, 281, 297) pass no sealed sets (observed-only seam) and assert only non-repeat gates.

---

### Task 2: Per-context rate-limit backoff — fail-fast in serving, long backoff offline, both fallback branches pinned

**Files:**
- Modify: `src/adapters/secondary/llm/groq_openai_client.py:252-260` (`__init__`), `:447-488` (`_get_provider_response`); `src/adapters/primary/http/app.py:54` (lifespan wiring)
- Test: `tests/test_llm_adapter.py` (reuse `_settings`, `_run`, `_async_client_with_create`, `_groq_rate_limit_error`, `VALID_JSON_TEXT`, `VALID_PAYLOAD`, `JSON_SCHEMA`, patch decorators); `tests/test_http_app_startup.py` (reuse `app_module`, TestClient pattern)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `GroqOpenAiLlmClient(..., rate_limit_backoff_seconds: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS)`; lifespan constructs the serving client with `rate_limit_backoff_seconds=()` and default `allow_provider_fallback=True`. Task 5 consumes this constructor shape.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_llm_adapter.py`:

```python
@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_fail_fast_backoff_raises_without_sleeping_when_fallback_disabled(
    mock_groq_cls, mock_openai_cls, mock_sleep
):
    mock_groq_cls.return_value = _async_client_with_create(_groq_rate_limit_error())

    with pytest.raises(GenerationError):
        _run(
            GroqOpenAiLlmClient(
                allow_provider_fallback=False, rate_limit_backoff_seconds=()
            ).generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))
        )

    assert mock_groq_cls.return_value.chat.completions.create.await_count == 1
    mock_sleep.assert_not_called()
    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.asyncio.sleep", new_callable=AsyncMock)
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_fail_fast_backoff_tries_fallback_immediately_without_sleeping(
    mock_groq_cls, mock_openai_cls, mock_sleep
):
    mock_groq_cls.return_value = _async_client_with_create(_groq_rate_limit_error())
    mock_openai_cls.return_value = _async_client_with_create(_response(VALID_JSON_TEXT))

    result = _run(
        GroqOpenAiLlmClient(rate_limit_backoff_seconds=()).generate_structured(
            "system", "user", JSON_SCHEMA, _settings("groq")
        )
    )

    assert result == VALID_PAYLOAD
    assert mock_groq_cls.return_value.chat.completions.create.await_count == 1
    assert mock_openai_cls.return_value.chat.completions.create.await_count == 1
    mock_sleep.assert_not_called()
```

The pair pins the verdict's open question explicitly: fail-fast removes sleeps only — with serving's default fallback enabled, a 429 still tries the second provider immediately; with fallback disabled it raises at once.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_adapter.py -q -k "fail_fast"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'rate_limit_backoff_seconds'` on both.

- [ ] **Step 3: Write minimal implementation** — extend `__init__`:

```python
    def __init__(
        self,
        *,
        allow_provider_fallback: bool = True,
        trace_hook: Optional[TraceHook] = None,
        rate_limit_backoff_seconds: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        """...`rate_limit_backoff_seconds=()` means fail-fast: one physical
        attempt per provider, no sleep. Serving wires fail-fast (a user-facing
        query must not sleep 105s behind nginx/httpx 60s timeouts) while keeping
        provider fallback; offline evaluation keeps the default long schedule."""
        self._allow_provider_fallback = allow_provider_fallback
        self._trace_hook = trace_hook
        self._rate_limit_backoff_seconds = tuple(rate_limit_backoff_seconds)
```

and in `_get_provider_response` replace the three uses of the module constant with `self._rate_limit_backoff_seconds`:

```python
        for attempt_index in range(len(self._rate_limit_backoff_seconds) + 1):
            ...
                if attempt_index >= len(self._rate_limit_backoff_seconds):
                    ...
                if wait_seconds is None:
                    wait_seconds = self._rate_limit_backoff_seconds[attempt_index]
```

Then wire serving fail-fast in `src/adapters/primary/http/app.py:54`:

```python
    llm_client = GroqOpenAiLlmClient(trace_hook=log_llm_trace, rate_limit_backoff_seconds=())
```

The offline eval factory in `gate_generation_eval.py:917-920` is intentionally untouched — its default keeps `(15, 30, 60)`, already pinned by existing tests `test_rate_limit_backs_off_via_asyncio_sleep_and_retries_same_provider` and `test_rate_limit_backoff_exhausted_falls_over_to_secondary`, which construct the default client.

- [ ] **Step 4: Write the wiring test** — append to `tests/test_http_app_startup.py` (module is already index-gated, mirroring existing lifespan tests):

```python
def test_lifespan_wires_fail_fast_llm_backoff(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    real_client_cls = app_module.GroqOpenAiLlmClient

    def _recording(*args: object, **kwargs: object):
        captured.update(kwargs)
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(app_module, "GroqOpenAiLlmClient", _recording)

    with TestClient(app_module.create_app()):
        pass

    assert captured.get("rate_limit_backoff_seconds") == ()
```

- [ ] **Step 5: Run the touched files' tests**

Run: `pytest tests/test_llm_adapter.py tests/test_http_app_startup.py -q`
Expected: PASS, no existing test changed (default constructor behavior is byte-identical).

---

### Task 3: CORS allows X-Client-Session

**Files:**
- Modify: `src/adapters/primary/http/app.py:96`
- Test: `tests/test_http_app_startup.py` (reuse `app_module`, TestClient pattern; env pattern from `tests/test_core_config.py:77`)

**Interfaces:**
- Consumes: nothing.
- Produces: preflight `access-control-allow-headers` containing `x-client-session` when CORS is enabled.

- [ ] **Step 1: Write the failing test**:

```python
def test_cors_preflight_allows_client_session_header(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://example.com")

    with TestClient(app_module.create_app()) as client:
        response = client.options(
            "/query",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Client-Session",
            },
        )

    assert response.status_code == 200
    assert "x-client-session" in response.headers["access-control-allow-headers"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_http_app_startup.py::test_cors_preflight_allows_client_session_header -q`
Expected: FAIL — allowed headers are `content-type, x-api-key` only.

- [ ] **Step 3: Write minimal implementation** — `src/adapters/primary/http/app.py:96`:

```python
            allow_headers=["Content-Type", "X-API-Key", "X-Client-Session"],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_http_app_startup.py -q`
Expected: PASS.

---

### Task 4: Incomplete prompt metadata degrades to canonical refusal

**Files:**
- Modify: `src/domain/models.py:17-32` (additive `DecisionReason` literal), `src/features/query/use_cases.py` (guard after line 160, before line 162)
- Test: `tests/test_query_use_case.py` (reuse `_result`, `_settings`, `InMemoryLLMClient`, `InMemoryRetriever`, `REFUSAL_MESSAGE`; `RetrievalResult` already imported at line 9)

**Interfaces:**
- Consumes: `RetrievalResult.metadata` dicts; `REFUSAL_MESSAGE` (already imported in use_cases).
- Produces: `QueryAnswer(refused=True, status="ok", decision_reason="incomplete_retrieved_metadata")` with the canonical refusal text whenever a top-5 prompt chunk lacks any of the five prompt-required fields as a non-empty `str`. `decision_reason` is logged only, never serialized (HTTP schema has no such field), so this is additive. The log `extra` carries request/event/language/band identifiers only — never chunk text, answer text, or citation payloads.

- [ ] **Step 1: Write the failing test** — append to `tests/test_query_use_case.py`:

```python
def test_incomplete_prompt_metadata_degrades_to_canonical_refusal():
    good = _result("chunk-1", 1, THRESHOLD + 0.2)
    metadata = dict(good.metadata)
    del metadata["chunk_text"]
    bad = RetrievalResult(
        chunk_id=good.chunk_id,
        fused_score=good.fused_score,
        semantic_rank=good.semantic_rank,
        semantic_score=good.semantic_score,
        bm25_rank=good.bm25_rank,
        bm25_score=good.bm25_score,
        metadata=metadata,
    )
    llm = InMemoryLLMClient(
        response={
            "answer": "The QC unit is responsible for X.",
            "citations": [{"chunk_id": "chunk-1"}],
            "refused": False,
        }
    )
    use_case = QueryUseCase(InMemoryRetriever([bad]), llm, _settings())

    answer = _run(use_case.answer_question("What is the QC unit responsible for?", "en"))

    assert answer.refused is True
    assert answer.status == "ok"
    assert answer.answer == REFUSAL_MESSAGE["en"]
    assert answer.citations == []
    assert answer.decision_reason == "incomplete_retrieved_metadata"
```

`chunk_text` is deliberately the removed field: the prompt needs it (`prompts.py:187`) but `CitationResolver` does not, so the test isolates the new guard instead of tripping citation resolution.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_use_case.py::test_incomplete_prompt_metadata_degrades_to_canonical_refusal -q`
Expected: ERROR with `KeyError: 'chunk_text'` propagating out of `answer_question` — whose contract is to always return a `QueryAnswer`, never raise for bad metadata. (A propagating `KeyError` is the valid RED demonstration here.)

- [ ] **Step 3: Write minimal implementation** — add the literal to the `DecisionReason` union in `src/domain/models.py`:

```python
    "unresolved_citation",
    "incomplete_retrieved_metadata",
    "accepted_grounded",
```

and insert the guard in `_answer_question_impl` after the hard-refuse block returns (after line 160) and before the prompt is built (line 162), validating the five prompt-required fields jointly as non-empty strings:

```python
        prompt_metadata_incomplete = any(
            not isinstance(result.metadata.get(field), str) or not result.metadata.get(field, "").strip()
            for result in prompt_results
            for field in (
                "document_title",
                "section_heading",
                "revision",
                "source_type",
                "chunk_text",
            )
        )
        if prompt_metadata_incomplete:
            logger.info(
                "prompt context metadata incomplete; refusing",
                extra={
                    "request_id": request_id,
                    "event": "incomplete_prompt_metadata_downgraded_to_refusal",
                    "language": language,
                    "gate_band": gate_band,
                },
            )
            answer = self._refusal_answer(
                language=language,
                confidence=score,
                gate_band=gate_band,
                decision_reason="incomplete_retrieved_metadata",
                request_id=request_id,
            )
            _log_query_completed(
                request_id, True, "ok", gate_band, "incomplete_retrieved_metadata", score, start_time
            )
            return answer
```

No other resolver changes: downstream `CitationResolver`/`GroundedEvidenceResolver` semantics are untouched; this guard only converts a 500 into the canonical refusal earlier.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_query_use_case.py -q`
Expected: PASS — including `test_normal_answer_path_resolves_citations_from_retrieved_metadata` (complete metadata flows through the guard untouched).

---

### Task 5: Provider-keyed reusable SDK clients, idempotent close, single-loop runners

**Files:**
- Modify: `src/adapters/secondary/llm/groq_openai_client.py` (`__init__` from Task 2, `_call_groq`, `_call_openai`, new `_sdk_client` + `aclose`, new `hashlib` import); `src/adapters/primary/http/app.py` (try opens immediately after line 54); `src/features/evaluation/gate_generation_eval.py:316-338` (`run_matrix` owns one loop); `src/features/evaluation/generation_eval.py:48-55,116-155,259-264` (async row builder, owned-client close)
- Test: `tests/test_llm_adapter.py`; `tests/test_http_app_startup.py`; `tests/test_evaluation_gate_generation_eval.py` (reuse `_settings`); `tests/test_evaluation_generation_eval.py` (reuse `_write_fake_eval_set`, `_build_fixtures`, `_FakeHeader`, patched `time.sleep` + `resolve_provenance` patterns from line 234)

**Interfaces:**
- Consumes: Task 2's `__init__` signature (`allow_provider_fallback`, `trace_hook`, `rate_limit_backoff_seconds`).
- Produces: one cached SDK client per provider keyed by key-fingerprint (never the secret); `async def aclose(self) -> None`, idempotent; serving + both offline runners create, use and close clients inside a single owning event loop.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_llm_adapter.py`:

```python
@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_sdk_client_reused_across_calls_and_closed_by_aclose(mock_groq_cls, mock_openai_cls):
    groq_client = _async_client_with_create(_response(VALID_JSON_TEXT))
    groq_client.close = AsyncMock()
    mock_groq_cls.return_value = groq_client

    client = GroqOpenAiLlmClient()
    _run(client.generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))
    _run(client.generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))

    assert mock_groq_cls.call_count == 1
    assert groq_client.chat.completions.create.await_count == 2
    _run(client.aclose())
    groq_client.close.assert_awaited_once()
    mock_openai_cls.assert_not_called()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_aclose_is_idempotent_and_safe_without_calls(mock_groq_cls, mock_openai_cls):
    groq_client = _async_client_with_create(_response(VALID_JSON_TEXT))
    groq_client.close = AsyncMock()
    mock_groq_cls.return_value = groq_client

    fresh = GroqOpenAiLlmClient()
    _run(fresh.aclose())
    _run(fresh.aclose())
    mock_groq_cls.assert_not_called()

    used = GroqOpenAiLlmClient()
    _run(used.generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))
    _run(used.aclose())
    _run(used.aclose())
    groq_client.close.assert_awaited_once()


@patch("src.adapters.secondary.llm.groq_openai_client.openai.AsyncOpenAI")
@patch("src.adapters.secondary.llm.groq_openai_client.groq.AsyncGroq")
def test_credential_change_rebuilds_provider_client_without_storing_secret(mock_groq_cls, mock_openai_cls):
    first_client = _async_client_with_create(_response(VALID_JSON_TEXT))
    first_client.close = AsyncMock()
    second_client = _async_client_with_create(_response(VALID_JSON_TEXT))
    second_client.close = AsyncMock()
    mock_groq_cls.side_effect = [first_client, second_client]
    rotated = Settings(
        groq_api_key="rotated-key",
        openai_api_key="openai-test-key",
        llm_provider="groq",
        refusal_cosine_threshold=0.5599,
        log_level="INFO",
    )

    client = GroqOpenAiLlmClient()
    _run(client.generate_structured("system", "user", JSON_SCHEMA, _settings("groq")))
    _run(client.generate_structured("system", "user", JSON_SCHEMA, rotated))

    assert mock_groq_cls.call_count == 2
    first_client.close.assert_awaited_once()
    second_client.close.assert_not_called()
    assert "rotated-key" not in repr(client._clients)
    assert "groq-test-key" not in repr(client._clients)
    _run(client.aclose())
    second_client.close.assert_awaited_once()
```

(`Settings` is already imported in that file at line 16.) The cache is keyed by provider with only a sha256 fingerprint beside the client — raw key material never becomes a long-lived dict entry.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_adapter.py -q -k "aclose or credential_change or reused"`
Expected: FAIL — `mock_groq_cls.call_count == 2` after two same-key calls (per-call construction) and `AttributeError` on `aclose` (method does not exist).

- [ ] **Step 3: Write minimal implementation** — add `import hashlib` to the module imports; in `__init__` add:

```python
        self._clients: dict[str, tuple[Optional[str], Any]] = {}
        self._lock = asyncio.Lock()
```

(`asyncio.Lock()` takes no loop argument on 3.10+, so constructing it in `__init__` is loop-safe; it only ever contends inside the single owning loop of serving or one runner.) Add the accessor and teardown (`Any` and `Optional` are already imported in that module):

```python
    @staticmethod
    def _key_fingerprint(api_key: Optional[str]) -> Optional[str]:
        """sha256 of the key, never the key itself: the cache lives as long
        as the process and must not become a long-lived copy of secrets."""
        if api_key is None:
            return None
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    async def _sdk_client(self, provider: str, api_key: Optional[str]) -> Any:
        """One SDK client per provider, reused across calls. No `await`
        sits between the hit-path lookup and return, and rotation is
        serialized by the lock; a changed key rebuilds that provider's
        client, closing the stale one first, so rotation inside one
        instance cannot serve with stale credentials."""
        want = self._key_fingerprint(api_key)
        async with self._lock:
            entry = self._clients.get(provider)
            if entry is not None:
                cached_fp, cached_client = entry
                if cached_fp == want:
                    return cached_client
                await cached_client.close()
            if provider == "groq":
                created: Any = groq.AsyncGroq(api_key=api_key, max_retries=0)
            else:
                created = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
            self._clients[provider] = (want, created)
            return created

    async def aclose(self) -> None:
        """Lifespan/runner teardown. Idempotent: closing twice, or closing a
        client that never made a call, closes nothing and raises nothing."""
        async with self._lock:
            cached = list(self._clients.values())
            self._clients.clear()
        for _, cached_client in cached:
            await cached_client.close()
```

Replace the two per-call constructions: in `_call_groq`, `client = groq.AsyncGroq(api_key=api_key, max_retries=0)` becomes `client = await self._sdk_client("groq", api_key)`; in `_call_openai`, likewise `await self._sdk_client("openai", api_key)` (both callers are already `async`).

Documented limitation (not supported): concurrent credential rotation. The lock serializes cache mutation, but a rotation could still close a client while another in-flight request uses it. Serving passes a single stable `Settings` from lifespan and both runners are sequential, so the rotation path never triggers in production — out of P1 scope, no blocking impact.

- [ ] **Step 4: Single owning loop per runner + lifespan teardown placed before state assignment.** In `src/adapters/primary/http/app.py`, open the `try` immediately after line 54 (NOT after `RateLimiter` — otherwise a `RateLimiter` construction failure skips teardown of the already-created client):

```python
    retriever = HybridRetriever(vector_store, lexical_index)
    llm_client = GroqOpenAiLlmClient(trace_hook=log_llm_trace, rate_limit_backoff_seconds=())

    try:
        app.state.settings = settings
        app.state.vector_store = vector_store
        app.state.query_use_case = QueryUseCase(retriever, llm_client, settings)
        app.state.rate_limiter = RateLimiter(max_requests=settings.rate_limit_per_minute)

        yield
    finally:
        await llm_client.aclose()
```

Serving is safe for caching: uvicorn runs one event loop for the process lifetime, and the SDK constructors bind nothing — first use inside that loop binds the connections there. In `gate_generation_eval.py::run_matrix`, replace the per-question `asyncio.run` loop with one owning loop (call order repeat → policy → question and the per-repeat `TraceCollector`/`WithinRepeatCache` are preserved verbatim; fakes without `aclose` are skipped by the guard; `asyncio` is already imported at line 4):

```python
def run_matrix(
    questions: list[dict[str, Any]],
    replay: ReplayRetriever,
    settings: Settings,
    llm_factory: Callable[[TraceHook], LLMClientPort],
    *,
    repeats: int,
) -> list[QuestionOutcome]:
    """One owning event loop per call: each per-repeat LLM client is created,
    used and closed inside this same loop. A cached httpx-based SDK client
    must never cross `asyncio.run` boundaries — its connections stay bound
    to the loop that opened them."""

    async def _run_all() -> list[QuestionOutcome]:
        outcomes: list[QuestionOutcome] = []
        for repeat in range(1, repeats + 1):
            trace = TraceCollector()
            llm = llm_factory(trace)
            cache = WithinRepeatCache(llm)
            try:
                for policy in _POLICIES:  # binary first so the confident call is cached
                    use_case = _use_case(policy, replay, cache, settings)
                    for question in questions:
                        outcomes.append(
                            await _run_question(
                                use_case, cache, trace, repeat=repeat, policy=policy, question=question
                            )
                        )
            finally:
                maybe_close = getattr(llm, "aclose", None)
                if callable(maybe_close):
                    await maybe_close()
        return outcomes

    return asyncio.run(_run_all())
```

In `generation_eval.py`, convert `_build_row` (line 116 — verified by grep to have no test callers) to `async def _build_row_async` with `answer = await use_case.answer_question(...)` replacing the `asyncio.run(...)` call (keep the leading `time.sleep(INTER_QUESTION_DELAY_SECONDS)` with a comment that the sequential runner blocks its owned loop and has no concurrency to starve); change `_build_use_case_and_retriever` (lines 48–55 — verified by grep to have no test callers) to return the owned client:

```python
def _build_use_case_and_retriever(
    expansion_mode: ExpansionMode = "off",
    index_profile: IndexProfile = "raw-v1",
) -> tuple[QueryUseCase, HybridRetriever, GroqOpenAiLlmClient]:
    settings = load_settings()
    retriever = build_retriever(expansion_mode, expected_profile=index_profile)
    llm_client = GroqOpenAiLlmClient()
    return QueryUseCase(retriever, llm_client, settings), retriever, llm_client
```

and drive `run()` (lines 259–264) from one owning loop that also closes what it owns (injected fakes keep `owned_client = None`, so nothing caller-owned is ever closed):

```python
    owned_client: Optional[GroqOpenAiLlmClient] = None
    if use_case is None or retriever is None:
        assert_live_index_profile(index_profile)
        use_case, retriever, owned_client = _build_use_case_and_retriever(expansion_mode, index_profile)

    async def _collect_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for question in questions:
                rows.append(await _build_row_async(use_case, retriever, question))
        finally:
            if owned_client is not None:
                await owned_client.aclose()
        return rows

    rows: list[dict[str, Any]] = asyncio.run(_collect_rows())
```

(`Optional` is already imported at line 10; `asyncio` at line 3.)

- [ ] **Step 5: Write the lifecycle tests.** Lifespan partial-startup (append to `tests/test_http_app_startup.py`; `_raise` helper and `pytest` import already exist there; module is index-gated):

```python
def test_lifespan_closes_llm_client_on_partial_startup(monkeypatch: pytest.MonkeyPatch):
    closed: list[bool] = []
    real_cls = app_module.GroqOpenAiLlmClient

    class _Recording(real_cls):  # type: ignore[misc]
        async def aclose(self) -> None:
            closed.append(True)
            await super().aclose()

    monkeypatch.setattr(app_module, "GroqOpenAiLlmClient", _Recording)
    monkeypatch.setattr(app_module, "RateLimiter", _raise("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        with TestClient(app_module.create_app()):
            pass

    assert closed == [True]
```

`RateLimiter` is constructed at line 59, after the client at line 54 inside the new `try`, so the boom path must close the client — against the v2 placement this test would fail with `closed == []`. Offline per-repeat close (append to `tests/test_evaluation_gate_generation_eval.py`, reusing its `_settings`):

```python
def test_run_matrix_closes_llm_client_each_repeat():
    closed: list[int] = []

    def factory(hook):
        class _Fake:
            async def generate_structured(self, sp, up, schema, settings):
                return {"answer": "x", "citations": [], "refused": True}

            async def aclose(self) -> None:
                closed.append(1)

        return _Fake()

    questions = [{"id": "q1", "question": "q?", "language": "en", "answerable": False}]
    replay = gge.ReplayRetriever({"q?": []})
    gge.run_matrix(questions, replay, _settings(), factory, repeats=2)

    assert closed == [1, 1]
```

(`ReplayRetriever({"q?": []})` yields no results → `hard_refuse`, no LLM traffic — the close path is what is exercised; existing `run_matrix` tests at lines 179–194 use aclose-less fakes and pin the skip-guard.) Owned-client close in `generation_eval` (append to `tests/test_evaluation_generation_eval.py`, reusing its `_write_fake_eval_set`, `_build_fixtures`, `_FakeHeader`, and the patched `time.sleep` + `resolve_provenance` pattern from `test_run_writes_report_and_csv_with_expected_metrics` at line 236):

```python
@patch("src.features.evaluation.generation_eval.artifacts.resolve_provenance")
@patch("src.features.evaluation.generation_eval.time.sleep")
def test_run_closes_owned_llm_client(mock_sleep, mock_provenance, tmp_path, monkeypatch):
    import src.features.evaluation.generation_eval as gen_eval

    mock_provenance.return_value = _FakeHeader()
    eval_set_path = tmp_path / "fake_eval_set.json"
    _write_fake_eval_set(eval_set_path)
    use_case, retriever = _build_fixtures()

    closed: list[bool] = []

    class _RecordingClient:
        async def aclose(self) -> None:
            closed.append(True)

    recording = _RecordingClient()
    monkeypatch.setattr(gen_eval, "assert_live_index_profile", lambda profile: None)
    monkeypatch.setattr(
        gen_eval,
        "_build_use_case_and_retriever",
        lambda *a, **k: (use_case, retriever, recording),
    )

    gen_eval.run(eval_set_path=eval_set_path, report_dir=tmp_path / "reports")

    assert closed == [True]
```

(`patch` is already imported in that file per its line-234 usage; `assert_live_index_profile` is patched so the test needs no built index.)

- [ ] **Step 6: Run the touched files' tests**

Run: `pytest tests/test_llm_adapter.py tests/test_http_app_startup.py tests/test_evaluation_gate_generation_eval.py tests/test_evaluation_generation_eval.py -q`
Expected: PASS — existing tests construct one client per test (single cache entry, behavior identical), assert on `mock_*_cls.return_value` (instance-level, unaffected by constructor deduplication), and the pre-existing `run_matrix` tests (lines 179–194) run unchanged through the single owning loop in identical repeat → policy → question order.

---

## Final Verification (all tasks, still no commits)

- [ ] Run: `pytest -q` — Expected: all pass (baseline was 522 passed).
- [ ] Run: `ruff check src tests` — Expected: all checks pass.
- [ ] Run: `mypy src` — Expected: no issues in 61 source files.
- [ ] Confirm no frozen file was re-stamped: `git status --short` shows only `src/**`, `tests/**`, and this plan file as modified/untracked; `eval/*.json`, `eval/reports/*_v1.*` untouched.

## Process Steps (no código, requieren confirmación explícita del owner)

- **P6 — Pilot untracked (#6):** `eval/reports/gate_generation_eval_20260902T225839Z/` (~300 KB: outcomes, retrieval, checklist, checksums, manifest, comparison) sustenta a `docs/eval/gate-generation-pilot-20260902.md`. Decidir `git add` del directorio o `.gitignore` explícito + nota. Es una operación irreversible-publicadora (el repo es público): no ejecutar sin `CONFIRM` explícito.
- **P7 — Higiene (#7):** `.test-tmp/` (warnings de permiso en `git status`; ningún código de `src/`/`tests/` lo crea — probable scratch de agentes), `.agents/`, `.commandcode/` untracked. Decidir limpiar vs ignorar+documentar. Tampoco ejecutar sin confirmación.
