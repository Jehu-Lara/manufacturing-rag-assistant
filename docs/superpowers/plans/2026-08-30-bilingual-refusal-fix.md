# Bilingual / terse-query false-refusal fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the RAG assistant from refusing questions it has the source material to answer — starting with the reported `NPSHA`/`NPSHR` queries (en + es) — via deterministic query expansion, measured against a frozen enlarged bilingual eval set.

**Architecture:** This plan covers **Phase 0 (frozen eval basis + preconditions)** and **Phase 1 (diagnose + measure C1, deterministic query-term expansion)** from the design doc. It ends at the C1 decision gate. Phases 2–4 (contextual embedding, gate recalibration, corpus addition) are conditional on Phase 1's measured results and get their own plan afterward. Nothing here changes production retrieval behaviour by default: query expansion ships behind an `expansion_mode` switch that defaults to `"off"`.

**Tech Stack:** Python 3.11, pytest, `sentence-transformers` (`BAAI/bge-m3`), `chromadb`, `rank_bm25`, `tiktoken`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-design.md` — read it first. The plan argues from the spec; executors read both.

## Global Constraints

- **Python 3.11** is the production/CI target. `mypy src` runs `--strict` (`python_version = "3.12"` is a permanent mypy-only setting — do not touch). `ruff check src tests` must pass.
- **`src/domain/` imports nothing framework-specific** — no fastapi/chromadb/groq/openai/streamlit/torch. Enforced by `tests/test_import_invariants.py`. `expand_query` and `GLOSSARY` live in `src/domain/policies.py` and must stay import-clean (`re` and stdlib only).
- **Byte-stable invariants — do NOT change in this plan:** `REFUSAL_COSINE_THRESHOLD = 0.5999` (`src/core/config.py`), RRF `k=60` + ascending-`chunk_id` tie-break (`src/domain/policies.py`).
- **No new module-level mutable singleton.** The compiled glossary regex is a module-level *immutable* constant (like `_TOKEN_PATTERN` in `bm25_lexical_index.py`) — that is allowed; a mutable cache is not.
- **Citation integrity unchanged:** citations are still re-derived from retrieved-chunk metadata by `chunk_id` (`CitationResolver`). This plan does not touch that path.
- **No real LLM calls in tests.** Use port fakes from `tests/fakes.py` or local stubs; never patch `answer_question`/`retrieve` at module level.
- **`eval/eval_set.json` is data, never packaged under `src/`.** Any content edit requires a `version` bump and `python -m src.features.evaluation.eval_set_integrity --write`.
- **`eval/reports/*_v1.0.0.md` are immutable** — never regenerate or edit them. All new output is `*_v1.1.0.md` with a provenance header.
- **Commits:** Conventional Commits format. Work on a branch (`git checkout -b fix/bilingual-refusal` before Task 1 — never commit to `master` directly). Commit after every task. **Do not push, open a PR, or deploy** — stop at the Phase 1 decision gate and report.
- **Corpus:** this plan adds **no** corpus files. If Phase 4 is later triggered, `corpus/SOURCES.md` rules apply then.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tests/test_http_contract_snapshot.py` | clear `API_KEY` so the snapshot test is env-independent | 1 |
| `tests/test_http_endpoints.py` | clear `API_KEY` so the 500-path test is env-independent | 1 |
| `eval/regression_queries.json` | frozen working set of terse/acronym/variant/control queries | 2 |
| `src/features/evaluation/regression_set_integrity.py` | SHA-256 freeze guard for the regression set (mirrors `eval_set_integrity`) | 2 |
| `tests/test_evaluation_regression_set_integrity.py` | verifies the regression set is frozen + well-formed | 2 |
| `eval/eval_set_v1.0.0.json` | archival frozen copy of the v1.0.0 eval set | 3 |
| `eval/eval_set.json` | evolved to v1.1.0 (≥25 es answerable + matched en + ≥15 es-relevant unanswerable) | 3 |
| `tests/test_evaluation_eval_set_integrity.py` | updated counts + bilingual assertions for v1.1.0 | 3 |
| `src/domain/models.py` | `ExpansionMode` literal alias | 4 |
| `src/domain/policies.py` | `GLOSSARY`, `expand_query()` | 4 |
| `tests/test_domain_policies.py` | `expand_query` + glossary tests | 4 |
| `src/features/retrieval/use_cases.py` | `HybridRetriever` applies `expand_query` per `expansion_mode` | 5 |
| `tests/test_hybrid_retriever_use_case.py` | spy stubs + per-mode expansion tests | 5 |
| `src/features/evaluation/threshold_analysis.py` | per-language + per-split sweep tables | 6 |
| `src/features/evaluation/_eval_retriever.py` | shared retriever builder taking `expansion_mode` | 7 |
| `src/features/evaluation/retrieval_eval.py` | accept `expansion_mode`; matched-pair cosine gap | 7 |
| `src/features/evaluation/regression_eval.py` | run the regression set in every config, write `regression_eval_v1.1.0.md` | 7 |
| `eval/reports/*_v1.1.0.md` | generated measurement output (provenance headers) | 8 |
| `docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md` | companion results doc: classification table + decision | 8 |

---

## Task 1: `API_KEY` test-isolation fix (required precondition)

**Files:**
- Modify: `tests/test_http_contract_snapshot.py` (add `API_KEY` cleanup to the fixture/test that exercises `/query`)
- Modify: `tests/test_http_endpoints.py:180-200` (`test_query_unhandled_exception_returns_500_with_generic_body`)

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: a green `pytest` baseline when a local `.env` with `API_KEY` is present.

**Context:** `src/adapters/primary/http/router.py:85-88` — when `settings.api_key is not None`, `/query` returns 401 before the use case runs. `load_settings()` reads `API_KEY` from the process env / `.env`. These two tests assume no key is configured. Other tests already neutralise ambient state with `app.dependency_overrides` (e.g. `get_rate_limiter`). Use the same mechanism: override `get_settings` (or `monkeypatch.delenv`) so `api_key` is `None` for these tests.

- [ ] **Step 1: Reproduce the failure**

Run: `python -m pytest tests/test_http_contract_snapshot.py::test_new_app_matches_phase0_snapshot tests/test_http_endpoints.py::test_query_unhandled_exception_returns_500_with_generic_body -q`
Expected: 2 failed (`assert 401 == 500`, snapshot mismatch at `query_unhandled_exception_500`) **if** a local `.env` sets `API_KEY`. If it passes, note "env has no API_KEY — fix still applied for isolation" and continue.

- [ ] **Step 2: Inspect both tests and the deps module**

Read `tests/test_http_endpoints.py` around the failing test, `tests/test_http_contract_snapshot.py` (the `_new_app_query_*` helpers), and `src/adapters/primary/http/deps.py` for `get_settings`.

- [ ] **Step 3: Add settings override to `test_http_endpoints.py`**

In `test_query_unhandled_exception_returns_500_with_generic_body`, alongside the existing `app.dependency_overrides[get_rate_limiter] = ...` line, add:

```python
from src.adapters.primary.http.deps import get_settings
from src.core.config import load_settings

_no_key_settings = load_settings().model_copy(update={"api_key": None})
app.dependency_overrides[get_settings] = lambda: _no_key_settings
```

(`Settings` is a frozen pydantic `BaseModel`; `model_copy(update=...)` is the supported way to derive one. Keep it inside the test, before the `TestClient` context, and it is cleared by the existing `app.dependency_overrides.clear()` in the `finally`.)

- [ ] **Step 4: Apply the same override in `test_http_contract_snapshot.py`**

Find the helper that builds the `query_unhandled_exception_500` case (and any other `_new_app_query_*` helper that posts to `/query` without a key). Add the same `get_settings` override around the `TestClient` call. If the helpers share a context-manager/fixture, put it there once.

- [ ] **Step 5: Run both tests**

Run: `python -m pytest tests/test_http_contract_snapshot.py::test_new_app_matches_phase0_snapshot tests/test_http_endpoints.py::test_query_unhandled_exception_returns_500_with_generic_body -q`
Expected: 2 passed.

- [ ] **Step 6: Full suite + lint + types**

Run: `python -m pytest -q && ruff check src tests && mypy src`
Expected: all pass (184 → 186 or same count, 0 failed). If `mypy` flags the `model_copy` line, add a targeted annotation, not an `ignore`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_http_contract_snapshot.py tests/test_http_endpoints.py
git commit -m "test(http): isolate /query tests from ambient API_KEY"
```

---

## Task 2: Frozen regression query set + integrity guard

**Files:**
- Create: `eval/regression_queries.json`
- Create: `src/features/evaluation/regression_set_integrity.py`
- Create: `tests/test_evaluation_regression_set_integrity.py`

**Interfaces:**
- Consumes: `ingestion/output/chunks.jsonl` (for `expected_chunk_id` validation).
- Produces:
  - `regression_set_integrity.load_regression_set(path=REGRESSION_SET_FILE) -> dict[str, Any]`
  - `regression_set_integrity.verify(path=REGRESSION_SET_FILE) -> None`
  - `regression_set_integrity.write(path=REGRESSION_SET_FILE) -> None`
  - `regression_set_integrity.REGRESSION_SET_FILE: Path`
  - `regression_queries.json` shape: `{"version": "1.0.0", "sha256": "...", "queries": [ {"id","query","language","expected_chunk_id"|null,"should_answer": bool,"note"} ]}`

- [ ] **Step 1: Write `regression_queries.json` with the seed rows**

Create `eval/regression_queries.json` with `"sha256": ""` for now and these queries (ids stable, `expected_chunk_id` verified in Step 3):

```json
{
  "version": "1.0.0",
  "sha256": "",
  "queries": [
    {"id": "r001", "query": "What is the difference between NPSHA and NPSHR?", "language": "en", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "reported live failure (en)"},
    {"id": "r002", "query": "¿Cuál es la diferencia entre NPSHA y NPSHR?", "language": "es", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "reported live failure (es)"},
    {"id": "r003", "query": "What is the difference between net positive suction head available (NPSHA) and net positive suction head required (NPSHR)?", "language": "en", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "acronym + inline expansion (en)"},
    {"id": "r004", "query": "¿Cuál es la diferencia entre la altura neta de succión positiva disponible (NPSHA) y la requerida (NPSHR)?", "language": "es", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "acronym + inline expansion (es)"},
    {"id": "r005", "query": "What is the difference between net positive suction head available and net positive suction head required?", "language": "en", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "expansion only, no acronym (en)"},
    {"id": "r006", "query": "What is NPSHR?", "language": "en", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "single acronym (en)"},
    {"id": "r007", "query": "¿Qué es NPSHR?", "language": "es", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "single acronym (es)"},
    {"id": "r008", "query": "How is cavitation avoided in a centrifugal pump?", "language": "en", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "terse, no acronym (en)"},
    {"id": "r009", "query": "¿Qué es la cavitación en una bomba centrífuga y qué la causa?", "language": "es", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0006", "should_answer": true, "note": "eval q009, gate-refused (es)"},
    {"id": "r010", "query": "What is the OSHA PEL for acetone?", "language": "en", "expected_chunk_id": "niosh-pocket-guide-excerpt::chunk-0010", "should_answer": true, "note": "acronym + decoy front-matter (en)"},
    {"id": "r011", "query": "¿Cuál es el PEL de la acetona?", "language": "es", "expected_chunk_id": "niosh-pocket-guide-excerpt::chunk-0010", "should_answer": true, "note": "acronym + decoy front-matter (es)"},
    {"id": "r012", "query": "What is the IDLH for carbon monoxide?", "language": "en", "expected_chunk_id": "niosh-pocket-guide-excerpt::chunk-0018", "should_answer": true, "note": "acronym + decoy front-matter (en)"},
    {"id": "r013", "query": "¿Cuál es el IDLH del monóxido de carbono?", "language": "es", "expected_chunk_id": "niosh-pocket-guide-excerpt::chunk-0018", "should_answer": true, "note": "acronym + decoy front-matter (es)"},
    {"id": "r014", "query": "What are the six pieces of information on a hazardous chemical label under HazCom?", "language": "en", "expected_chunk_id": "cfr-29-1910-1200-hazcom::chunk-0008", "should_answer": true, "note": "abbreviation 'HazCom' (en)"},
    {"id": "r015", "query": "Can an employer be exempt from LOTO if machine guarding removes exposure?", "language": "en", "expected_chunk_id": "osha-3170-machine-guarding::chunk-0021", "should_answer": true, "note": "acronym LOTO, cross-doc (en)"},
    {"id": "r016", "query": "¿Un empleador puede quedar exento de LOTO si el resguardo elimina la exposición?", "language": "es", "expected_chunk_id": "osha-3170-machine-guarding::chunk-0021", "should_answer": true, "note": "acronym LOTO, cross-doc (es)"},
    {"id": "r017", "query": "NPSH-A vs NPSHa", "language": "en", "expected_chunk_id": "doe-hdbk-1018-1-pumps::chunk-0007", "should_answer": true, "note": "surface variants / hyphenation (en)"},
    {"id": "r018", "query": "What NPSH margin does API 610 recommend for this pump?", "language": "en", "expected_chunk_id": null, "should_answer": false, "note": "control: API 610 not in corpus (en)"},
    {"id": "r019", "query": "¿Qué margen de NPSH recomienda la norma API 610 para esta bomba?", "language": "es", "expected_chunk_id": null, "should_answer": false, "note": "control: API 610 not in corpus (es)"},
    {"id": "r020", "query": "What is the PEL for toluene diisocyanate?", "language": "en", "expected_chunk_id": null, "should_answer": false, "note": "control: TDI not among the 12 NIOSH excerpt chemicals (en)"}
  ]
}
```

- [ ] **Step 2: Write `regression_set_integrity.py`**

Copy the structure of `src/features/evaluation/eval_set_integrity.py` exactly, changing:
- `EVAL_SET_FILE` → `REGRESSION_SET_FILE = .../"eval"/"regression_queries.json"`
- hash over `data["queries"]` (not `data["questions"]`)
- `load_eval_set` → `load_regression_set`
- error message references `regression_set_integrity --write`

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

REGRESSION_SET_FILE = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "regression_queries.json"


def canonical_queries_bytes(queries: list[dict[str, Any]]) -> bytes:
    return json.dumps(queries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_hash(queries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_queries_bytes(queries)).hexdigest()


def load_regression_set(path: Path = REGRESSION_SET_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def verify(path: Path = REGRESSION_SET_FILE) -> None:
    data = load_regression_set(path)
    actual = compute_hash(data["queries"])
    if actual != data["sha256"]:
        raise ValueError(
            f"{path} sha256 mismatch — stored {data['sha256']}, computed {actual}. "
            "If intentional, bump 'version' and re-run "
            "`python -m src.features.evaluation.regression_set_integrity --write`."
        )


def write(path: Path = REGRESSION_SET_FILE) -> None:
    data = load_regression_set(path)
    data["sha256"] = compute_hash(data["queries"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print(f"{REGRESSION_SET_FILE}: hash OK")
    else:
        write()
        print(f"{REGRESSION_SET_FILE}: hash regenerated")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_evaluation_regression_set_integrity.py`:

```python
from __future__ import annotations

import json

from src.features.evaluation.eval_set_integrity import EVAL_SET_FILE
from src.features.evaluation.regression_set_integrity import (
    REGRESSION_SET_FILE,
    load_regression_set,
    verify,
)

_CHUNKS_FILE = EVAL_SET_FILE.resolve().parent.parent / "ingestion" / "output" / "chunks.jsonl"


def _known_chunk_ids() -> set[str]:
    with _CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f}


def test_regression_set_hash_is_frozen():
    verify()


def test_regression_rows_are_well_formed():
    data = load_regression_set()
    ids = [q["id"] for q in data["queries"]]
    assert len(ids) == len(set(ids))
    known = _known_chunk_ids()
    for q in data["queries"]:
        assert q["language"] in ("en", "es")
        assert isinstance(q["should_answer"], bool)
        if q["should_answer"]:
            assert q["expected_chunk_id"] in known, f"{q['id']}: {q['expected_chunk_id']!r} not in chunks.jsonl"
        else:
            assert q["expected_chunk_id"] is None


def test_regression_set_has_language_pairs_and_controls():
    data = load_regression_set()
    assert sum(1 for q in data["queries"] if q["language"] == "es") >= 6
    assert sum(1 for q in data["queries"] if not q["should_answer"]) >= 2
```

- [ ] **Step 4: Run — expect hash failure**

Run: `python -m pytest tests/test_evaluation_regression_set_integrity.py -q`
Expected: `test_regression_set_hash_is_frozen` FAILS (stored `""` ≠ computed). The other two may pass or fail on chunk-id mismatch.

- [ ] **Step 5: Fix any bad `expected_chunk_id`, then freeze the hash**

Run: `python -c "import json; f='ingestion/output/chunks.jsonl'; ids={json.loads(l)['chunk_id'] for l in open(f,encoding='utf-8')}; [print(x) for x in sorted(i for i in ids if 'pumps' in i or 'niosh' in i or 'hazcom' in i or 'machine-guarding' in i)]"`
Cross-check every `expected_chunk_id` in `regression_queries.json` against that list. Fix mismatches (the NIOSH acetone/CO and pumps NPSH chunk numbers in Step 1 are best-effort — verify them). Then:

Run: `python -m src.features.evaluation.regression_set_integrity --write`

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_evaluation_regression_set_integrity.py -q`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add eval/regression_queries.json src/features/evaluation/regression_set_integrity.py tests/test_evaluation_regression_set_integrity.py
git commit -m "test(eval): add frozen bilingual regression query set"
```

---

## Task 3: eval_set v1.1.0 — freeze v1.0.0, enlarge Spanish coverage

**Files:**
- Create: `eval/eval_set_v1.0.0.json` (byte copy of current `eval/eval_set.json`)
- Modify: `eval/eval_set.json` (→ v1.1.0)
- Modify: `tests/test_evaluation_eval_set_integrity.py`

**Interfaces:**
- Consumes: `ingestion/output/chunks.jsonl`, the corpus under `corpus/`.
- Produces: `eval_set.json` with `version == "1.1.0"`, ≥ 55 answerable questions (30 original + ≥ 25 new, of which ≥ 25 are `es` and each new `es` has an `en` counterpart with the same `expected_chunk_ids`), ≥ 25 unanswerable (10 original `en` + ≥ 15 `es`).

**Context:** The existing 40 questions in `eval/eval_set.json` stay **unchanged** (same ids, text, expected chunks). New questions get ids `q041`+. Every answerable question's `expected_chunk_ids` must exist in `chunks.jsonl` (enforced by `test_every_answerable_question_has_expected_chunk_ids_that_exist`). Question authoring is a judgement task — follow the method below, do not invent chunk ids.

- [ ] **Step 1: Archive v1.0.0**

```bash
cp eval/eval_set.json eval/eval_set_v1.0.0.json
git add eval/eval_set_v1.0.0.json
```

This file is never loaded by code — it is the immutable record. Confirm `git status` shows it staged.

- [ ] **Step 2: List candidate chunks per corpus domain**

Run: `python -c "import json; [print(json.loads(l)['chunk_id'], '|', json.loads(l)['metadata']['section_heading'][:70]) for l in open('ingestion/output/chunks.jsonl',encoding='utf-8')]" | sort`

Use this map to pick `expected_chunk_ids` for new questions. Cover all domains: LOTO, machine guarding, PPE, pumps, valves, DC theory, NIOSH chemicals, 21 CFR 211, 29 CFR 1910.1200, and the 5 synthetic docs.

- [ ] **Step 3: Author the new Spanish answerable questions (≥ 25) + English counterparts**

For each: pick a real section, read its chunk text, write a natural Spanish question a plant worker would type (mix terse, acronym, and full phrasings — **not** all fully-spelled-out), then the identical-meaning English question. Both point at the **same** `expected_chunk_ids`. Schema per question (match the existing rows exactly):

```json
{
  "id": "q041",
  "question": "¿Cada cuánto se lubrican los rodamientos del transportador XJ-450?",
  "language": "es",
  "answerable": true,
  "expected_chunk_ids": ["manual-xj450-belt-conveyor::chunk-0004"],
  "expected_document_id": "manual-xj450-belt-conveyor",
  "expected_section_heading": "3. Preventive Maintenance Schedule > ...",
  "expected_answer": "Mensualmente, según el programa de lubricación.",
  "notes": "es terse phrasing; en counterpart q042"
},
{
  "id": "q042",
  "question": "How often are the XJ-450 conveyor bearings lubricated?",
  "language": "en",
  "answerable": true,
  "expected_chunk_ids": ["manual-xj450-belt-conveyor::chunk-0004"],
  "expected_document_id": "manual-xj450-belt-conveyor",
  "expected_section_heading": "3. Preventive Maintenance Schedule > ...",
  "expected_answer": "Monthly, per the lubrication schedule.",
  "notes": "en counterpart of q041"
}
```

Include at least these acronym-bearing pairs so C1 has signal: `NPSHA`/`NPSHR` (pumps chunk-0007), `PEL`/`IDLH`/`TWA` (NIOSH), `SDS` (hazcom chunk-0009), `PPE` (osha-3151), `LOTO` (osha-3120 / osha-3170), `CGMP` (cfr-21-211). Reuse the exact acronyms in the frozen `GLOSSARY` (Task 4).

- [ ] **Step 4: Author the new Spanish unanswerable questions (≥ 15)**

Spanish versions of genuinely-out-of-corpus topics (respirator fit-testing, confined space, arc-flash PPE, forklift certification, bloodborne pathogens, portable fire extinguishers, mechanical power presses, three-phase AC motor faults, audiometric testing, plus new ones: welding ventilation rates, crane rigging, machine risk assessment ISO 12100, etc.). Each:

```json
{
  "id": "q0NN",
  "question": "¿Con qué frecuencia deben inspeccionarse los extintores portátiles?",
  "language": "es",
  "answerable": false,
  "expected_chunk_ids": [],
  "expected_document_id": null,
  "expected_section_heading": null,
  "expected_answer": "No answerable from this corpus — the correct system behavior is to decline rather than guess.",
  "notes": "29 CFR 1910.157 not in corpus (es)"
}
```

- [ ] **Step 5: Bump version + regenerate hash**

Set `"version": "1.1.0"` in `eval/eval_set.json`. Then:

Run: `python -m src.features.evaluation.eval_set_integrity --write`

- [ ] **Step 6: Update the integrity tests for v1.1.0**

In `tests/test_evaluation_eval_set_integrity.py`:
- `test_eval_set_has_forty_items_with_declared_split` → rename to `test_eval_set_split_counts` and assert the new totals (compute them: `len(answerable) == <N>`, `len(unanswerable) == <M>`, `len(questions) == N+M`). Keep the assertion exact, not `>=`, so future drift is caught.
- `test_spanish_questions_present_for_bilingual_validation` → assert `len(spanish_answerable) >= 25` and `len(spanish_unanswerable) >= 15`; drop the `all(q["answerable"])` assertion (Spanish now spans both splits) and the `<= 8` upper bound.
- Add `test_every_new_spanish_answerable_has_english_counterpart`: for each `es` answerable question with id ≥ `q041`, assert some `en` answerable question shares its `expected_chunk_ids` (set-equal).

- [ ] **Step 7: Run integrity + full suite**

Run: `python -m pytest tests/test_evaluation_eval_set_integrity.py -q`
Expected: all pass.
Run: `python -m pytest -q && ruff check src tests`
Expected: 0 failed. (Retrieval/generation eval tests that hard-code v1.0.0 numbers, if any, will fail here — fix them to read from `load_eval_set()` or mark the specific numeric assertion as v1.0.0-historical. List any you change.)

- [ ] **Step 8: Commit**

```bash
git add eval/eval_set.json eval/eval_set_v1.0.0.json tests/test_evaluation_eval_set_integrity.py
git commit -m "test(eval): enlarge Spanish coverage (eval_set v1.1.0), archive v1.0.0"
```

---

## Task 4: `expand_query` + `GLOSSARY` in the domain layer

**Files:**
- Modify: `src/domain/models.py` (add `ExpansionMode`)
- Modify: `src/domain/policies.py` (add `GLOSSARY`, `_GLOSSARY_PATTERN`, `expand_query`)
- Modify: `tests/test_domain_policies.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `src.domain.models.ExpansionMode = Literal["off", "semantic", "lexical", "both"]`
  - `src.domain.policies.GLOSSARY: dict[str, tuple[str, ...]]`
  - `src.domain.policies.expand_query(query: str) -> str` — returns `query` unchanged if no glossary key matches; else `query + " " + " ".join(additions)` with additions in `GLOSSARY` insertion order then tuple order, de-duplicated, skipping any expansion already present (case-insensitive substring) in `query`.

- [ ] **Step 1: Add `ExpansionMode` to `models.py`**

Next to `Language = Literal["en", "es"]` (line 6):

```python
ExpansionMode = Literal["off", "semantic", "lexical", "both"]
```

- [ ] **Step 2: Write the failing tests in `test_domain_policies.py`**

Add:

```python
from src.domain.policies import GLOSSARY, expand_query


def test_expand_query_passthrough_when_no_glossary_key():
    q = "What is the frame level tolerance for the conveyor?"
    assert expand_query(q) == q


def test_expand_query_appends_expansions_for_known_acronym():
    out = expand_query("What is the difference between NPSHA and NPSHR?")
    assert out.startswith("What is the difference between NPSHA and NPSHR?")
    assert "net positive suction head available" in out
    assert "net positive suction head required" in out
    assert "altura neta de succión positiva disponible" in out


def test_expand_query_is_case_insensitive_and_word_bounded():
    assert "net positive suction head available" in expand_query("what is npsha")
    unchanged = "the NPSHATEST rig and xNPSHA probe"
    assert expand_query(unchanged) == unchanged


def test_expand_query_multi_term_order_is_deterministic():
    a = expand_query("PEL and IDLH for acetone")
    b = expand_query("IDLH and PEL for acetone")
    # additions ordered by GLOSSARY insertion order, not query order → identical tails
    assert a.split("for acetone", 1)[1] == b.split("for acetone", 1)[1]
    assert expand_query("PEL and IDLH for acetone") == a  # stable across calls


def test_expand_query_dedupes_expansion_already_present():
    q = "define permissible exposure limit PEL"
    out = expand_query(q)
    assert out.count("permissible exposure limit") == 1


def test_glossary_english_expansions_are_corpus_attested():
    import pathlib

    corpus_text = " ".join(
        p.read_text(encoding="utf-8").lower()
        for p in pathlib.Path("corpus").rglob("*.md")
    )
    for key, expansions in GLOSSARY.items():
        english = expansions[0]
        assert english.lower() in corpus_text, f"{key}: {english!r} not found in corpus"


def test_glossary_spanish_renderings_nonempty_and_distinct():
    for key, expansions in GLOSSARY.items():
        assert len(expansions) >= 2, f"{key}: needs an es rendering"
        es = expansions[1]
        assert es.strip()
        assert es.lower() != expansions[0].lower()
```

- [ ] **Step 3: Run — expect ImportError / failures**

Run: `python -m pytest tests/test_domain_policies.py -q`
Expected: FAIL (`cannot import name 'GLOSSARY'`).

- [ ] **Step 4: Implement in `policies.py`**

After the imports and `RRF_K` (top of file), add:

```python
import re

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
    "IDLH": ("immediately dangerous to life and health", "concentración inmediatamente peligrosa para la vida o la salud"),
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
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_domain_policies.py -q`
Expected: all pass. If `test_glossary_english_expansions_are_corpus_attested` fails for a key, grep the corpus for the real phrasing and fix `expansions[0]` (or drop that key — keep ≥ 8).

- [ ] **Step 6: Import-invariant + lint + types**

Run: `python -m pytest tests/test_import_invariants.py -q && ruff check src tests && mypy src`
Expected: all pass (domain stays framework-free; `re` is stdlib).

- [ ] **Step 7: Commit**

```bash
git add src/domain/models.py src/domain/policies.py tests/test_domain_policies.py
git commit -m "feat(retrieval): deterministic corpus-derived query acronym expansion"
```

---

## Task 5: Wire `expansion_mode` into `HybridRetriever`

**Files:**
- Modify: `src/features/retrieval/use_cases.py`
- Modify: `tests/test_hybrid_retriever_use_case.py`

**Interfaces:**
- Consumes: `expand_query` (Task 4), `ExpansionMode` (Task 4).
- Produces: `HybridRetriever(vector_store, lexical_index, expansion_mode: ExpansionMode = "off")`. When `"off"`, byte-identical behaviour to today. `"semantic"` → expanded query to the vector store only; `"lexical"` → to BM25 only; `"both"` → both.

- [ ] **Step 1: Write failing tests**

In `tests/test_hybrid_retriever_use_case.py`, add spy stubs and tests:

```python
class _SpyVectorStore(_StubVectorStore):
    def query(self, text: str, top_n: int):
        self.last_query = text
        return super().query(text, top_n)


class _SpyLexicalIndex(_StubLexicalIndex):
    def query(self, text: str, top_n: int):
        self.last_query = text
        return super().query(text, top_n)


def _spies():
    md = {"c": {"document_id": "d"}}
    vs = _SpyVectorStore(hits=[("c", 0.9, md["c"])], metadata_by_id=md)
    lx = _SpyLexicalIndex(hits=[("c", 1.0)])
    return vs, lx


def test_expansion_mode_off_passes_original_to_both():
    vs, lx = _spies()
    HybridRetriever(vs, lx).retrieve("What is NPSHA?", k=1)
    assert vs.last_query == "What is NPSHA?"
    assert lx.last_query == "What is NPSHA?"


def test_expansion_mode_semantic_expands_vector_only():
    vs, lx = _spies()
    HybridRetriever(vs, lx, expansion_mode="semantic").retrieve("What is NPSHA?", k=1)
    assert "net positive suction head available" in vs.last_query
    assert lx.last_query == "What is NPSHA?"


def test_expansion_mode_lexical_expands_bm25_only():
    vs, lx = _spies()
    HybridRetriever(vs, lx, expansion_mode="lexical").retrieve("What is NPSHA?", k=1)
    assert vs.last_query == "What is NPSHA?"
    assert "net positive suction head available" in lx.last_query


def test_expansion_mode_both_expands_both():
    vs, lx = _spies()
    HybridRetriever(vs, lx, expansion_mode="both").retrieve("What is NPSHA?", k=1)
    assert "net positive suction head available" in vs.last_query
    assert "net positive suction head available" in lx.last_query
```

- [ ] **Step 2: Run — expect failure**

Run: `python -m pytest tests/test_hybrid_retriever_use_case.py -q`
Expected: FAIL (`HybridRetriever() got unexpected keyword 'expansion_mode'`).

- [ ] **Step 3: Implement**

In `src/features/retrieval/use_cases.py`:

```python
from src.domain.models import ExpansionMode, RetrievalResult
from src.domain.policies import expand_query, fuse_rankings
```

```python
    def __init__(
        self,
        vector_store: VectorStorePort,
        lexical_index: LexicalIndexPort,
        expansion_mode: ExpansionMode = "off",
    ) -> None:
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._expansion_mode = expansion_mode
```

In `retrieve()`, replace the first two lines inside the span:

```python
            expanded = expand_query(query_text)
            semantic_query = expanded if self._expansion_mode in ("semantic", "both") else query_text
            lexical_query = expanded if self._expansion_mode in ("lexical", "both") else query_text
            semantic_hits = self._vector_store.query(semantic_query, top_n)
            bm25_hits = self._lexical_index.query(lexical_query, top_n)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_hybrid_retriever_use_case.py tests/test_core_telemetry.py -q`
Expected: all pass (the `"off"` default keeps the existing 3 stub tests + telemetry stub test green).

- [ ] **Step 5: Full suite + lint + types**

Run: `python -m pytest -q && ruff check src tests && mypy src`
Expected: 0 failed. Existing retrieval/generation eval reports still reproduce because production and all current call sites use `"off"`.

- [ ] **Step 6: Commit**

```bash
git add src/features/retrieval/use_cases.py tests/test_hybrid_retriever_use_case.py
git commit -m "feat(retrieval): expansion_mode switch on HybridRetriever (default off)"
```

---

## Task 6: Per-language / per-split threshold analysis

**Files:**
- Modify: `src/features/evaluation/threshold_analysis.py`
- Modify (if present): `tests/test_evaluation_threshold_analysis.py`

**Interfaces:**
- Consumes: `load_eval_set()` (v1.1.0), `HybridRetriever`.
- Produces: `threshold_analysis_v1.1.0.md` with, in addition to today's pooled sweep: a sweep table filtered to `language == "es"` and one to `language == "en"`, plus the answerable/unanswerable score lists split by language. No behavioural change; no threshold selection change.

- [ ] **Step 1: Read the current module + its test**

Read `src/features/evaluation/threshold_analysis.py` and any `tests/test_evaluation_threshold_analysis.py`.

- [ ] **Step 2: Write/extend the failing test**

Add a test that calls `build_report(...)` with synthetic per-language rows and asserts the output contains the headings `## English — cutoff sweep` and `## Spanish — cutoff sweep`. If a helper needs the question language, thread it through (`run()` already has `q["language"]`).

- [ ] **Step 3: Implement**

Change `run()` to keep `(score, language)` pairs, and `build_report()` to emit, after the pooled section:
- `## English — answerable/unanswerable top-1 semantic_score` + `## English — cutoff sweep`
- the same for Spanish
Reuse the existing `_stats_line` / sweep helpers, filtering the input lists by language. The **selection procedure and chosen threshold stay pooled and unchanged** — add a one-line note: "Per-language tables are diagnostic; the shipped threshold remains the pooled selection (0.5999 override, SPEC.md Phase 3)."

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_evaluation_threshold_analysis.py -q`
Expected: pass.

- [ ] **Step 5: Lint + types**

Run: `ruff check src tests && mypy src`

- [ ] **Step 6: Commit**

```bash
git add src/features/evaluation/threshold_analysis.py tests/test_evaluation_threshold_analysis.py
git commit -m "feat(eval): per-language threshold sweep tables"
```

---

## Task 7: Measurement scripts (`_eval_retriever`, `regression_eval`, matched-pair gap)

**Files:**
- Create: `src/features/evaluation/_eval_retriever.py`
- Modify: `src/features/evaluation/retrieval_eval.py`, `threshold_analysis.py`, `generation_eval.py` (use the shared builder)
- Create: `src/features/evaluation/regression_eval.py`
- Create: `tests/test_evaluation_regression_eval.py`

**Interfaces:**
- Consumes: `HybridRetriever(..., expansion_mode=...)`, `load_regression_set()`, `load_eval_set()`, `metrics`.
- Produces:
  - `_eval_retriever.build_retriever(expansion_mode: ExpansionMode = "off") -> HybridRetriever`
  - `regression_eval.run(expansion_modes: list[ExpansionMode] = ["off"]) -> Path` — writes `eval/reports/regression_eval_v<eval_set_version>.md`
  - `retrieval_eval.run(expansion_mode: ExpansionMode = "off") -> Path` — unchanged default; report gains a `## Matched-pair cosine gap (en − es)` section.

- [ ] **Step 1: Extract the shared retriever builder**

Create `src/features/evaluation/_eval_retriever.py`:

```python
from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.models import ExpansionMode
from src.features.retrieval.use_cases import HybridRetriever


def build_retriever(expansion_mode: ExpansionMode = "off") -> HybridRetriever:
    settings = load_settings()
    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)
    return HybridRetriever(vector_store, lexical_index, expansion_mode=expansion_mode)
```

Replace the duplicated `_build_retriever()` in `retrieval_eval.py`, `threshold_analysis.py`, `generation_eval.py` with `from src.features.evaluation._eval_retriever import build_retriever` (keep their public `run()` signatures; pass `expansion_mode` through where they accept it).

- [ ] **Step 2: Write `regression_eval.py`**

```python
from __future__ import annotations

from pathlib import Path

from src.domain.models import ExpansionMode
from src.domain.policies import RefusalPolicy, top1_semantic_score_from_results
from src.features.evaluation import eval_set_integrity, regression_set_integrity
from src.features.evaluation._eval_retriever import build_retriever
from src.features.evaluation.metrics import recall_at_k
from src.features.retrieval.use_cases import SEMANTIC_EXTRACTION_K

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "reports"


def _row(retriever, q, threshold: float) -> dict:
    results = retriever.retrieve(q["query"], k=SEMANTIC_EXTRACTION_K)
    ids = [r.chunk_id for r in results]
    score = top1_semantic_score_from_results(results)
    confident = RefusalPolicy(threshold).is_confident(results)
    hit5 = recall_at_k(ids, [q["expected_chunk_id"]], 5) if q["expected_chunk_id"] else None
    return {"id": q["id"], "language": q["language"], "should_answer": q["should_answer"],
            "top1_semantic": score, "gate": "answer" if confident else "REFUSE",
            "recall@5": hit5, "top1_chunk": ids[0] if ids else None}


def run(expansion_modes: list[ExpansionMode] | None = None) -> Path:
    modes = expansion_modes or ["off"]
    regression_set_integrity.verify()
    eval_set_integrity.verify()
    data = regression_set_integrity.load_regression_set()
    version = eval_set_integrity.load_eval_set()["version"]
    threshold = 0.5999  # byte-stable invariant; diagnostic only

    lines = [f"# Regression Eval — eval_set v{version}", "",
             f"- threshold (diagnostic): {threshold}", ""]
    for mode in modes:
        retriever = build_retriever(mode)
        rows = [_row(retriever, q, threshold) for q in data["queries"]]
        lines += [f"## expansion_mode = {mode}", "",
                  "| id | lang | should_answer | top1_semantic | gate | recall@5 |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['id']} | {r['language']} | {r['should_answer']} | "
                         f"{r['top1_semantic']:.4f} | {r['gate']} | {r['recall@5']} |")
        answered_ok = sum(1 for r in rows if r["should_answer"] and r["gate"] == "answer")
        refused_ok = sum(1 for r in rows if not r["should_answer"] and r["gate"] == "REFUSE")
        n_ans = sum(1 for r in rows if r["should_answer"])
        n_ref = sum(1 for r in rows if not r["should_answer"])
        lines += ["", f"- answerable passing gate: {answered_ok}/{n_ans}",
                  f"- controls correctly refused: {refused_ok}/{n_ref}", ""]

    report = REPORT_DIR / f"regression_eval_v{version}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to: {report}")
    return report


if __name__ == "__main__":
    run(["off", "semantic", "lexical", "both"])
```

- [ ] **Step 3: Write a light test**

`tests/test_evaluation_regression_eval.py` — use a scripted fake retriever (per `tests/test_evaluation_generation_eval.py`'s pattern) so no model loads; assert `_row` classifies gate/recall correctly and `run(["off"])` writes a file with the `## expansion_mode = off` heading.

- [ ] **Step 4: Add the matched-pair gap to `retrieval_eval.py`**

In `build_report`, after the per-language recall lines, add a section: for each `en` answerable question that shares `expected_chunk_ids` with an `es` answerable question, record `en_top1_semantic - es_top1_semantic`; print the per-pair values and the mean. (Pair by set-equal `expected_chunk_ids`; if multiple candidates, pair by nearest id.)

- [ ] **Step 5: Run tests + lint + types**

Run: `python -m pytest tests/test_evaluation_regression_eval.py tests/test_evaluation_retrieval_eval.py -q && ruff check src tests && mypy src`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/features/evaluation/_eval_retriever.py src/features/evaluation/regression_eval.py src/features/evaluation/retrieval_eval.py src/features/evaluation/threshold_analysis.py src/features/evaluation/generation_eval.py tests/test_evaluation_regression_eval.py
git commit -m "feat(eval): regression_eval + shared retriever builder + matched-pair gap"
```

---

## Task 8: Run the measurements, generate v1.1.0 reports, decision gate

**Files:**
- Create: `eval/reports/retrieval_report_v1.1.0.md`, `threshold_analysis_v1.1.0.md`, `regression_eval_v1.1.0.md`, `generation_eval_v1.1.0.md` (generated)
- Create: `docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a filled-in acceptance table (spec §6) and a decision: ship a C1 config, or proceed to a Phase 2 plan.

**Context:** Each `run()` here loads `BAAI/bge-m3` and queries the built index (`retrieval/output/`). If the index predates any corpus change, rebuild first: `python -m src.features.ingestion.cli && python -m src.features.retrieval.cli` (~10 min CPU). This plan changed **no** corpus files, so the existing index is valid — but confirm `chunks.jsonl` still matches (`python -m src.features.evaluation.eval_set_integrity --verify` will not catch a stale index; a stale index shows as every recall@5 dropping).

- [ ] **Step 1: Baseline (`expansion_mode=off`) against v1.1.0**

Run:
```bash
python -m src.features.evaluation.retrieval_eval
python -m src.features.evaluation.threshold_analysis
python -m src.features.evaluation.generation_eval
python -m src.features.evaluation.regression_eval
```
These write `*_v1.1.0.md` (retrieval/threshold/generation) and `regression_eval_v1.1.0.md` (modes `off,semantic,lexical,both` — the `__main__` block runs all four). Add a provenance header line to each: git commit, eval_set v1.1.0 + hash, regression_set hash, date. **Do not touch any `*_v1.0.0.md`.**

- [ ] **Step 2: C1 per-config retrieval + generation runs**

For `mode in semantic lexical both`:
```bash
python -c "from src.features.evaluation.retrieval_eval import run; run('$mode')"
python -c "from src.features.evaluation.generation_eval import run; run(expansion_mode='$mode')"
```
Append each config's summary (Recall@3/@5 per language, MRR, matched-pair gap, correct-/false-refusal) into `retrieval_report_v1.1.0.md` / `generation_eval_v1.1.0.md` under a `## expansion_mode = <mode>` heading, or write `retrieval_report_v1.1.0_<mode>.md` — pick one and be consistent. (`generation_eval.run` may need an `expansion_mode` param plumbed through in Task 7 Step 1 — verify it accepts one; if not, add it now as a 1-line change and note it.)

- [ ] **Step 3: Classification table**

Create `docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md`. For every v1.1.0 answerable miss and every regression failure at `expansion_mode=off`, one row: `id | language | retrieval OK? | gate decision | class ∈ {gate-over-refusal, retrieval-miss, decoy-chunk}`.

- [ ] **Step 4: Fill the acceptance table (spec §6) per config**

| config | EN Recall@5 | ES Recall@5 | matched-pair gap | EN correct-refusal | ES correct-refusal | EN false-refusal | ES false-refusal | r001/r002 answered? |

- [ ] **Step 5: Decision**

- If a C1 config clears every **hard** cell of the acceptance table → record "SHIP `expansion_mode=<mode>`"; the follow-up is a tiny task to set the production default (config/env + `main.py`), written as its own plan increment.
- Else → record which cells fail and by how much; the follow-up is the **Phase 2 (contextual embedding) plan**, seeded with the classification table.
- Either way: **stop. Report to the owner. No push, no PR, no deploy, no threshold change.**

- [ ] **Step 6: Commit the reports + results doc**

```bash
git add eval/reports/*_v1.1.0*.md docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md
git commit -m "docs(eval): v1.1.0 measurements + C1 config comparison + decision"
```

- [ ] **Step 7: Final verification gate**

Run: `python -m pytest -q && ruff check src tests && mypy src && python -m src.features.evaluation.eval_set_integrity --verify && python -m src.features.evaluation.regression_set_integrity --verify`
Expected: all green.
Run: `git status && git diff --staged --stat` — confirm no `.env`, no API keys, no PDFs, no `corpus/` changes, no `*_v1.0.0.md` modifications.

---

## Self-Review

**Spec coverage:**
- §6 acceptance table → Task 8 Step 4. ✅
- §6 Phase 0 step 1 (API_KEY) → Task 1. ✅
- §6 Phase 0 step 2–3 (regression set + integrity + committed versioned report) → Task 2, Task 7. ✅
- §6 Phase 0 step 4–5 (eval v1.1.0, v1.0.0 immutable) → Task 3. ✅
- §6 Phase 1 step 6 (stratified threshold analysis) → Task 6. ✅
- §6 Phase 1 step 7 (classification) → Task 8 Step 3. ✅
- §6 Phase 1 step 8 (`expand_query`, two evidence tracks, deterministic order+dedup) → Task 4. ✅
- §6 Phase 1 step 9 (C1 measured sem/bm25/both) → Task 5 (switch), Task 7–8 (runs). ✅
- §6 Phase 1 step 10 (decision gate) → Task 8 Step 5. ✅
- §8 test table → Tasks 4, 5, 7 (each row mapped). ✅
- §6 Phases 2–4 → **explicitly out of this plan**, seeded by Task 8's results doc. ✅
- §6 Phase 5 verification → Task 8 Step 7. ✅

**Placeholder scan:** Task 3 (question authoring) gives method + schema + worked examples rather than 40 literal rows — this is inherent to eval-set authoring and matches how the existing set was built; not a code step. Task 8 Step 2 leaves a formatting choice (one file vs per-mode files) explicitly to the implementer with "pick one and be consistent". No `TBD`/`TODO`/"handle edge cases" strings. ✅

**Type consistency:** `ExpansionMode` defined in `models.py` (Task 4), consumed in `use_cases.py` (Task 5) and `_eval_retriever.py` (Task 7) with the same `Literal` members. `expand_query(query: str) -> str` signature identical across Tasks 4, 5, 7. `build_retriever(expansion_mode="off")` / `HybridRetriever(..., expansion_mode="off")` consistent. `load_regression_set` / `REGRESSION_SET_FILE` names consistent Tasks 2 & 7. ✅

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-30-bilingual-refusal-fix.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — tasks executed in this session with checkpoints for review.

**Which approach?**
