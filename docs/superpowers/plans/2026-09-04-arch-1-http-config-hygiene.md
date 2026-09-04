# Bucket 1 — HTTP / Config / Validation Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the structural leaks the audit confirmed — FastAPI inside `src/features/`, the serving path importing the index-build CLI, sixteen hand-rolled repo-root chains, three env vars read outside `load_settings()`, a module-level mutable singleton, and two input-validation holes — without moving any byte-stable invariant.

**Architecture:** Seven tasks. T1 moves the router into `src/adapters/primary/http/`, which is the single change that lets T7's AST invariant forbid `fastapi` under `src/features/`. T2 gives `load_chunks` its own module so `app.lifespan` stops importing a build CLI. T3 introduces `src/core/paths.py` as the one repo-root authority. T4 folds `INDEX_PROFILE`, `DEPLOYED_SHA` and the OTLP endpoint into `Settings`. T5 closes the blank-question and non-positive-limit holes. T6 replaces `telemetry._configured`. T7 seals the boundaries with AST tests and syncs the docs that currently assert the old shape. Ordering constraints: T1 → T7, T4 → T7, and T3 → T2 only if you want `chunk_store` to bind `CHUNKS_FILE` from `paths` on first write. Everything else is order-free.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2 (`field_validator`), pytest, `ast` (static invariants), opentelemetry-sdk.

**Spec:** `docs/superpowers/specs/2026-09-04-architecture-remediation-design.md`

## Global Constraints

- Execution is gated: PR #9 resolved by the owner, then a new branch cut from `master` with the owner's authorization. No commits, pushes, merges or deploys without a separate explicit request.
- `REFUSAL_COSINE_THRESHOLD` stays `0.5999`; `REFUSAL_REVIEW_FLOOR` stays `0.5500`; RRF `k=60` with ascending-`chunk_id` tie-break; `REFUSAL_POLICY` default stays `binary`; `expansion_mode` default stays `off`; served profile stays `contextual-v1`.
- Frozen datasets are never re-stamped; `eval/reports/*_v1.*` and `*__raw-v1__off.*` are never overwritten.
- No corpus content change, so no chunk-id re-anchoring is in scope.
- After each task: `pytest <touched-test-files> -q` green. At the end of the bucket: `pytest -q`, `ruff check src tests`, `mypy src` all green.
- Locally the venv is `.\.venv\Scripts\python.exe -m pytest`; CI uses bare `pytest`.

---

## File Structure

- `src/adapters/primary/http/routes.py` — **new.** Verbatim move of today's `src/features/query/router.py`. Owns `router`, `_to_response_schema`, `_rate_limit_key`, and the three endpoints. The only place outside `app.py`/`deps.py` that imports FastAPI.
- `src/features/query/router.py` — **deleted.** `src/features/query/` keeps `use_cases.py` and `prompts.py` and becomes framework-free.
- `src/features/retrieval/chunk_store.py` — **new.** Owns `CHUNKS_FILE` and `load_chunks`. Imports only stdlib + `src.core.paths` + `src.domain.models`; deliberately pulls in no adapter.
- `src/core/paths.py` — **new.** The one `REPO_ROOT` definition plus derived corpus/ingestion/retrieval/eval directories.
- `src/core/config.py` — gains `index_profile`, `deployed_sha`, `otlp_endpoint`; `load_settings()` reads and validates the three new env vars and rejects a non-positive `RATE_LIMIT_PER_MINUTE`.
- `src/core/telemetry.py` — `_configured` deleted; idempotence derived from the live tracer provider; endpoint becomes a parameter.
- `src/adapters/primary/http/schemas.py` — `QueryRequest` rejects whitespace-only questions.
- `src/adapters/primary/http/app.py` — imports the moved router and the new chunk store; passes the settings-supplied profile and OTLP endpoint.
- `src/features/retrieval/index_manifest.py`, `src/features/evaluation/*.py`, `src/features/ingestion/*.py` — path constants re-pointed at `src.core.paths`.
- `tests/test_import_invariants.py` — two new invariants. `tests/test_core_paths.py`, `tests/test_core_telemetry.py`, `tests/test_retrieval_chunk_store.py` — new.
- Docs: `CLAUDE.md`, `docs/adr/001-modular-monolith-src-layout.md`, `docs/architecture/` C4 sources.

---

### Task 1: Move the query router into the HTTP adapter

**Files:**
- Create: `src/adapters/primary/http/routes.py`
- Delete: `src/features/query/router.py`
- Modify: `src/adapters/primary/http/app.py:19`
- Test: `tests/test_import_invariants.py`

**Interfaces:**
- Consumes: `src.adapters.primary.http.deps.{get_query_use_case,get_rate_limiter,get_settings,get_vector_store}`, `src.adapters.primary.http.schemas.*`, `src.features.query.use_cases.QueryUseCase`.
- Produces: `src.adapters.primary.http.routes.router` (`APIRouter`) and `src.adapters.primary.http.routes._rate_limit_key(session_id, http_request)`. Every test or doc that imports `src.features.query.router` must be re-pointed.

- [ ] **Step 1: Write the failing test** — append to `tests/test_import_invariants.py`:

```python
FEATURES_ROOT = SRC_ROOT / "features"

# src/features/ orchestrates ports; it must never bind a web framework or
# reach into the primary (driving) adapter. The router lived here before the
# 2026-09-04 architecture remediation and was the sole violation.
FEATURES_FORBIDDEN_TOP_LEVEL = {"fastapi"}
FEATURES_FORBIDDEN_PACKAGES = ("src.adapters.primary",)


def test_features_layer_never_imports_fastapi_or_primary_adapters() -> None:
    violations: dict[str, set[str]] = {}
    for py_file in FEATURES_ROOT.rglob("*.py"):
        found = _imported_top_level_modules(py_file) & FEATURES_FORBIDDEN_TOP_LEVEL
        found |= {
            module
            for module in _imported_module_paths(py_file)
            for package in FEATURES_FORBIDDEN_PACKAGES
            if module == package or module.startswith(package + ".")
        }
        if found:
            violations[str(py_file)] = found
    assert not violations, f"src/features imported a web framework or a primary adapter: {violations}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_import_invariants.py::test_features_layer_never_imports_fastapi_or_primary_adapters -v`

Expected: FAIL, listing `src/features/query/router.py` with `{'fastapi', 'src.adapters.primary.http.deps', 'src.adapters.primary.http.rate_limit', 'src.adapters.primary.http.schemas'}`.

- [ ] **Step 3: Move the file**

```bash
git mv src/features/query/router.py src/adapters/primary/http/routes.py
```

Its contents need no edit: every import it already makes (`deps`, `rate_limit`, `schemas`, `sentence_transformers_embedder.MODEL_NAME`, `QueryUseCase`) is legal from inside the adapter layer. Then update `src/adapters/primary/http/app.py:19`:

```python
from src.adapters.primary.http.routes import router
```

- [ ] **Step 4: Re-point every remaining reference**

```bash
grep -rn "features\.query\.router\|features/query/router" src tests docs CLAUDE.md SPEC.md
```

Replace each hit with `src.adapters.primary.http.routes`. The docstring cross-reference in `src/adapters/primary/http/rate_limit.py:17-18` (`src.features.query.router._rate_limit_key`) is one of them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_import_invariants.py -q && pytest -q -k "router or endpoint or http or startup"`

Expected: PASS.

---

### Task 2: Give `load_chunks` its own module

**Files:**
- Create: `src/features/retrieval/chunk_store.py`
- Modify: `src/features/retrieval/cli.py:13-23`, `src/adapters/primary/http/app.py:22`, `src/features/evaluation/_eval_retriever.py:9`
- Test: `tests/test_retrieval_chunk_store.py` (new)

**Interfaces:**
- Produces: `src.features.retrieval.chunk_store.load_chunks(path: Path = CHUNKS_FILE) -> list[ChunkMetadata]` and `src.features.retrieval.chunk_store.CHUNKS_FILE`. `src.features.retrieval.cli` re-exports both by importing them, so `from src.features.retrieval.cli import load_chunks` keeps resolving for any straggler.

- [ ] **Step 1: Write the failing tests** — create `tests/test_retrieval_chunk_store.py`:

```python
from __future__ import annotations

import ast
import json
from pathlib import Path

from src.domain.models import ChunkMetadata
from src.features.retrieval.chunk_store import CHUNKS_FILE, load_chunks

MODULE = Path(__file__).resolve().parent.parent / "src" / "features" / "retrieval" / "chunk_store.py"


def _chunk_row() -> dict[str, object]:
    """Built from ChunkMetadata's real field list so a field rename fails here
    loudly instead of drifting into a TypeError inside the behaviour tests."""
    defaults: dict[str, object] = {
        "document_id": "doc-a",
        "document_title": "Doc A",
        "section_heading": "S1",
        "source_type": "synthetic",
        "source_url_or_note": "note",
        "source_page_range": None,
        "chunk_id": "doc-a::chunk-0000",
        "chunk_text": "body",
        "md_line_range": "1-2",
    }
    fields = set(ChunkMetadata.__dataclass_fields__)
    missing = fields - set(defaults)
    assert not missing, f"extend _chunk_row for new ChunkMetadata fields: {sorted(missing)}"
    return {k: v for k, v in defaults.items() if k in fields}


def test_chunk_store_imports_no_adapter() -> None:
    """Serving reads chunk ids at startup; it must not have to import the
    index-build CLI (and through it chromadb + sentence-transformers) to do it."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    forbidden = {
        m
        for m in modules
        if m.startswith("src.adapters")
        or m.split(".")[0] in {"chromadb", "sentence_transformers", "torch"}
    }
    assert not forbidden, f"chunk_store must stay adapter-free, imported: {forbidden}"


def test_load_chunks_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(_chunk_row()) + "\n", encoding="utf-8")

    chunks = load_chunks(path)

    assert [c.chunk_id for c in chunks] == ["doc-a::chunk-0000"]


def test_load_chunks_missing_file_names_the_ingestion_command(tmp_path: Path) -> None:
    try:
        load_chunks(tmp_path / "absent.jsonl")
    except FileNotFoundError as exc:
        assert "src.features.ingestion.cli" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_default_chunks_file_is_the_repo_ingestion_output() -> None:
    assert CHUNKS_FILE.parts[-3:] == ("ingestion", "output", "chunks.jsonl")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval_chunk_store.py -q`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.features.retrieval.chunk_store'`.

- [ ] **Step 3: Create the module**

```python
# src/features/retrieval/chunk_store.py
from __future__ import annotations

import json
from pathlib import Path

from src.core.paths import CHUNKS_FILE
from src.domain.models import ChunkMetadata

__all__ = ["CHUNKS_FILE", "load_chunks"]


def load_chunks(path: Path = CHUNKS_FILE) -> list[ChunkMetadata]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.features.ingestion.cli` first to produce it"
        )
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(ChunkMetadata(**json.loads(line)))
    return chunks
```

If Task 3 has not landed yet, temporarily use `CHUNKS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "ingestion" / "output" / "chunks.jsonl"` here and let Task 3 replace it with the `src.core.paths` import.

Then in `src/features/retrieval/cli.py` delete lines 13–23 and import instead:

```python
from src.features.retrieval.chunk_store import CHUNKS_FILE, load_chunks

__all__ = ["CHUNKS_FILE", "load_chunks", "run"]
```

The `__all__` keeps ruff from flagging the re-exports as unused imports.

- [ ] **Step 4: Re-point the two real consumers**

`src/adapters/primary/http/app.py:22` and `src/features/evaluation/_eval_retriever.py:9` both become:

```python
from src.features.retrieval.chunk_store import load_chunks
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval_chunk_store.py -q && pytest -q -k "startup or retrieval"`

Expected: PASS.

---

### Task 3: One repo-root authority — `src/core/paths.py`

**Files:**
- Create: `src/core/paths.py`
- Modify: `src/core/config.py:41-43,73`, `src/features/retrieval/index_manifest.py:15-18`, `src/features/retrieval/chunk_store.py`, `src/features/ingestion/cli.py:10`, `src/features/ingestion/use_cases.py:12`, and the eleven `src/features/evaluation/*.py` lines in the table below
- Test: `tests/test_core_paths.py` (new)

**Interfaces:**
- Produces: `src.core.paths.REPO_ROOT`, `.CORPUS_DIR`, `.INGESTION_OUTPUT_DIR`, `.CHUNKS_FILE`, `.RETRIEVAL_OUTPUT_DIR`, `.EVAL_DIR`, `.EVAL_REPORTS_DIR`. Every module keeps its existing public constant name (`CHUNKS_FILE`, `REPORT_DIR`, `CORPUS_ROOT`, `OUTPUT_DIR`, `MANIFEST_FILE`, …) and simply binds it from here — no caller changes anywhere.

- [ ] **Step 1: Write the failing tests** — create `tests/test_core_paths.py`:

```python
from __future__ import annotations

from pathlib import Path

from src.core import paths

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def test_repo_root_is_the_directory_holding_pyproject() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / "src").is_dir()


def test_derived_paths_hang_off_repo_root() -> None:
    assert paths.CORPUS_DIR == paths.REPO_ROOT / "corpus"
    assert paths.CHUNKS_FILE == paths.REPO_ROOT / "ingestion" / "output" / "chunks.jsonl"
    assert paths.RETRIEVAL_OUTPUT_DIR == paths.REPO_ROOT / "retrieval" / "output"
    assert paths.EVAL_REPORTS_DIR == paths.REPO_ROOT / "eval" / "reports"


def test_no_other_module_rolls_its_own_repo_root_chain() -> None:
    """`Path(__file__).resolve().parent.parent.parent[.parent]` is off by one the
    moment a module moves between package depths — exactly what the router move
    in Task 1 does. src/core/paths.py is the single place allowed to compute it."""
    offenders: dict[str, int] = {}
    for py_file in SRC_ROOT.rglob("*.py"):
        if py_file == SRC_ROOT / "core" / "paths.py":
            continue
        count = py_file.read_text(encoding="utf-8").count("parent.parent.parent")
        if count:
            offenders[str(py_file.relative_to(SRC_ROOT))] = count
    assert not offenders, f"modules computing their own repo root: {offenders}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_core_paths.py -q`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.paths'`.

- [ ] **Step 3: Create the module**

```python
# src/core/paths.py
from __future__ import annotations

from pathlib import Path

# The single repo-root authority. This file sits at src/core/paths.py, so
# parent.parent.parent lands on the repo root — one hop shallower than the
# src/features/**/x.py modules that used four hops. Counting hops from each
# module's own depth is what breaks silently when a module moves between
# package levels, so every other module binds its constants from here.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CORPUS_DIR = REPO_ROOT / "corpus"
INGESTION_OUTPUT_DIR = REPO_ROOT / "ingestion" / "output"
CHUNKS_FILE = INGESTION_OUTPUT_DIR / "chunks.jsonl"
RETRIEVAL_OUTPUT_DIR = REPO_ROOT / "retrieval" / "output"
EVAL_DIR = REPO_ROOT / "eval"
EVAL_REPORTS_DIR = EVAL_DIR / "reports"
```

- [ ] **Step 4: Re-point every constant**

`src/core/config.py` — replace lines 41–43 and 73:

```python
from src.core.paths import REPO_ROOT, RETRIEVAL_OUTPUT_DIR

_DEFAULT_CHROMA_PATH = RETRIEVAL_OUTPUT_DIR / "chroma"
_DEFAULT_BM25_PATH = RETRIEVAL_OUTPUT_DIR / "bm25_index.json"
```

and inside `load_settings()`: `env_path = REPO_ROOT / ".env"`.

`src/features/retrieval/index_manifest.py` — replace lines 15–18:

```python
from src.core.paths import CHUNKS_FILE, CORPUS_DIR, REPO_ROOT, RETRIEVAL_OUTPUT_DIR

MANIFEST_FILE = RETRIEVAL_OUTPUT_DIR / "index_manifest.json"
```

`REPO_ROOT`, `CHUNKS_FILE` and `CORPUS_DIR` keep their module-level names here because `resolve_build_commit` (lines 74, 81) and the default arguments of `chunks_sha256`, `corpus_sha256`, `build_manifest` and `verify` already reference them by those names.

The remaining rewrites, each replacing one `Path(__file__).resolve().parent.parent.parent.parent / ...` expression while keeping the left-hand constant name unchanged:

| File:line | Existing name | New right-hand side |
|---|---|---|
| `src/features/ingestion/cli.py:10` | `OUTPUT_DIR` | `INGESTION_OUTPUT_DIR` |
| `src/features/ingestion/use_cases.py:12` | `CORPUS_ROOT` | `CORPUS_DIR` |
| `src/features/evaluation/eval_set_integrity.py:12` | `EVAL_SET_FILE` | `EVAL_DIR / "eval_set.json"` |
| `src/features/evaluation/regression_set_integrity.py:12` | `REGRESSION_SET_FILE` | `EVAL_DIR / "regression_queries.json"` |
| `src/features/evaluation/failure_classification.py:11` | `REPORT_DIR` | `EVAL_REPORTS_DIR` |
| `src/features/evaluation/gate_generation_eval.py:37` | `REPORT_ROOT` | `EVAL_REPORTS_DIR` |
| `src/features/evaluation/gate_holdout_integrity.py:14` | `_REPO_ROOT` | `REPO_ROOT` |
| `src/features/evaluation/gate_holdout_profile.py:18` | `REPORT_DIR` | `EVAL_REPORTS_DIR` |
| `src/features/evaluation/generation_eval.py:20` | `REPORT_DIR` | `EVAL_REPORTS_DIR` |
| `src/features/evaluation/regression_eval.py:14` | `REPORT_DIR` | `EVAL_REPORTS_DIR` |
| `src/features/evaluation/retrieval_eval.py:16` | `REPORT_DIR` | `EVAL_REPORTS_DIR` |
| `src/features/evaluation/retrieval_eval.py:17` | `REPO_ROOT` | `REPO_ROOT` (import, drop the local chain) |
| `src/features/evaluation/threshold_analysis.py:12` | `REPORT_DIR` | `EVAL_REPORTS_DIR` |

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_core_paths.py -q
pytest -q
```

Expected: PASS. A path typo surfaces as a `FileNotFoundError` in the integrity or ingestion suites, not as a silent skip.

---

### Task 4: Fold `INDEX_PROFILE`, `DEPLOYED_SHA` and the OTLP endpoint into `Settings`

**Files:**
- Modify: `src/core/config.py` (new aliases, fields, reads), `src/features/retrieval/index_manifest.py:59-85`, `src/core/telemetry.py:18-36`, `src/adapters/primary/http/app.py:38,103`, `src/features/retrieval/cli.py:27-29`, `src/features/evaluation/_eval_retriever.py:25`
- Test: `tests/test_core_config.py`, `tests/test_retrieval_index_manifest.py`

**Interfaces:**
- Produces: `Settings.index_profile: IndexProfileName`, `Settings.deployed_sha: Optional[str]`, `Settings.otlp_endpoint: Optional[str]`; `index_manifest.resolve_index_profile(settings: Settings | None = None) -> IndexProfile`; `index_manifest.resolve_build_commit(explicit: str | None = None, *, settings: Settings | None = None) -> str`; `telemetry.configure_tracing(app: FastAPI, *, otlp_endpoint: str | None = None) -> None`. Passing `None` keeps today's behaviour of loading settings internally, so existing zero-arg callers and tests keep working.

**Documented convention change:** `CLAUDE.md` currently states `INDEX_PROFILE` "is **not** a `Settings` field (mirrors `expansion_mode`)". This task deliberately reverses that, and Task 7 rewrites the sentence. The rationale holds: `expansion_mode` is non-`Settings` precisely so production cannot override it, whereas `INDEX_PROFILE` is already env-overridable by design (it is the documented `raw-v1` rollback path). Moving it in removes a scattered read without granting any new override.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_core_config.py`:

```python
def test_index_profile_defaults_to_contextual_v1(monkeypatch) -> None:
    monkeypatch.delenv("INDEX_PROFILE", raising=False)
    assert load_settings().index_profile == "contextual-v1"


def test_index_profile_is_env_overridable_to_the_rollback_profile(monkeypatch) -> None:
    monkeypatch.setenv("INDEX_PROFILE", "raw-v1")
    assert load_settings().index_profile == "raw-v1"


def test_invalid_index_profile_raises_at_load(monkeypatch) -> None:
    monkeypatch.setenv("INDEX_PROFILE", "contextual-v2")
    with pytest.raises(ValueError, match="INDEX_PROFILE"):
        load_settings()


def test_deployed_sha_and_otlp_endpoint_default_to_none(monkeypatch) -> None:
    monkeypatch.delenv("DEPLOYED_SHA", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    settings = load_settings()
    assert settings.deployed_sha is None
    assert settings.otlp_endpoint is None


def test_deployed_sha_and_otlp_endpoint_are_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYED_SHA", "a" * 40)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    settings = load_settings()
    assert settings.deployed_sha == "a" * 40
    assert settings.otlp_endpoint == "http://collector:4317"


def test_config_index_profile_literal_matches_the_domain_one() -> None:
    """Two structurally identical Literals, deliberately not shared: src/core
    must not import src/domain. This pins them together so they cannot drift."""
    from typing import get_args

    from src.core.config import IndexProfileName
    from src.domain.models import IndexProfile

    assert set(get_args(IndexProfileName)) == set(get_args(IndexProfile))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_core_config.py -q -k "index_profile or deployed_sha or otlp"`

Expected: FAIL — `ImportError: cannot import name 'IndexProfileName'` and `AttributeError: 'Settings' object has no attribute 'index_profile'`.

- [ ] **Step 3: Add the aliases, fields and reads**

In `src/core/config.py`, beside the existing `LlmProvider` / `RefusalPolicyName` aliases:

```python
IndexProfileName = Literal["raw-v1", "contextual-v1"]

_VALID_INDEX_PROFILES: tuple[IndexProfileName, ...] = ("raw-v1", "contextual-v1")
_DEFAULT_INDEX_PROFILE: IndexProfileName = "contextual-v1"
```

On `Settings`:

```python
    index_profile: IndexProfileName = _DEFAULT_INDEX_PROFILE
    deployed_sha: Optional[str] = None
    otlp_endpoint: Optional[str] = None
```

In `load_settings()`, before the `return`:

```python
    index_profile_raw = os.environ.get("INDEX_PROFILE", _DEFAULT_INDEX_PROFILE)
    if index_profile_raw not in _VALID_INDEX_PROFILES:
        raise ValueError(
            f"INDEX_PROFILE must be one of {_VALID_INDEX_PROFILES}, got {index_profile_raw!r}"
        )
    index_profile: IndexProfileName = index_profile_raw  # type: ignore[assignment]

    deployed_sha = os.environ.get("DEPLOYED_SHA") or None
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None
```

and pass all three into the `Settings(...)` construction.

- [ ] **Step 4: Delegate the three former readers**

`src/features/retrieval/index_manifest.py` — delete `_VALID_INDEX_PROFILES` / `_DEFAULT_INDEX_PROFILE` and the `os.environ` read:

```python
def resolve_index_profile(settings: Settings | None = None) -> IndexProfile:
    """The env read now lives in load_settings(); this keeps the zero-arg call
    convenience while leaving one authority for the variable. The cast is safe:
    config's IndexProfileName and domain's IndexProfile are pinned equal by
    tests/test_core_config.py::test_config_index_profile_literal_matches_the_domain_one."""
    resolved = settings if settings is not None else load_settings()
    return cast("IndexProfile", resolved.index_profile)
```

and `resolve_build_commit` replaces its line-71 `os.environ.get("DEPLOYED_SHA")`:

```python
def resolve_build_commit(explicit: str | None = None, *, settings: Settings | None = None) -> str:
    if explicit:
        return explicit
    resolved = settings if settings is not None else load_settings()
    if resolved.deployed_sha:
        return resolved.deployed_sha
    ...  # DEPLOYED_SHA file, then `git rev-parse HEAD`, then "unknown" — unchanged
```

Delete `import os` from this module once both reads are gone.

`src/core/telemetry.py` takes the endpoint as a parameter (Task 6 rewrites the rest of this function):

```python
def configure_tracing(app: "FastAPI", *, otlp_endpoint: str | None = None) -> None:
```

with line 31's `endpoint = os.environ.get(...)` deleted, `if otlp_endpoint:` guarding the exporter block, and `import os` removed.

`src/adapters/primary/http/app.py` — line 38 becomes `profile = index_manifest.resolve_index_profile(settings)` and line 103 becomes `configure_tracing(app, otlp_endpoint=settings.otlp_endpoint)`. `create_app` already calls `load_settings()` at line 79, so no extra read is introduced.

`src/features/retrieval/cli.py` — move `settings = load_settings()` above the profile line and call `index_manifest.resolve_index_profile(settings)`.

`src/features/evaluation/_eval_retriever.py:25` — `index_manifest.resolve_index_profile(settings)`, reusing the `settings` already loaded on line 24.

- [ ] **Step 5: Run the tests and confirm the env reads are centralized**

```bash
pytest tests/test_core_config.py tests/test_retrieval_index_manifest.py -q
pytest -q -k "startup or manifest or profile"
grep -rn "os.environ\|os.getenv" src/ --include=*.py
```

Expected: PASS, and the grep returns only `src/core/config.py` and `src/web/client.py` — `src/web/` is a separate process by design (ADR-005/007) and keeps its own two reads.

---

### Task 5: Reject blank questions with 422 and non-positive limits at startup

**Files:**
- Modify: `src/adapters/primary/http/schemas.py:10-12`, `src/core/config.py` (rate-limit block, lines 129–136)
- Test: the module that exercises `POST /query` (find it with `grep -rln '"/query"' tests/`), `tests/test_core_config.py`

**Interfaces:**
- Produces: `QueryRequest` rejects a question that is empty after `str.strip()` and stores the stripped value; `load_settings()` raises `ValueError` for `RATE_LIMIT_PER_MINUTE <= 0`.

- [ ] **Step 1: Write the failing tests**

In the `/query` endpoint test module (reuse whatever `client` / API-key fixtures it already defines; drop the `headers=` argument if that suite runs without an API key configured):

```python
@pytest.mark.parametrize("blank", ["   ", "\t", "\n", "   "])
def test_whitespace_only_question_is_rejected_with_422(client, blank: str) -> None:
    """min_length=1 accepts "   ": the retriever would then embed whitespace and
    the refusal gate would score noise. A blank question is a malformed request,
    not a question that happens to be unanswerable."""
    response = client.post("/query", json={"question": blank, "language": "en"})
    assert response.status_code == 422


def test_question_is_stored_stripped(client) -> None:
    response = client.post("/query", json={"question": "  what is a PQ?  ", "language": "en"})
    assert response.status_code == 200
```

In `tests/test_core_config.py`:

```python
@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_rate_limit_raises_at_load(monkeypatch, value: str) -> None:
    """A limiter with max_requests <= 0 rejects every request: the container
    boots, reports healthy, and answers 429 to everyone. Fail at startup."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", value)
    with pytest.raises(ValueError, match="RATE_LIMIT_PER_MINUTE"):
        load_settings()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_core_config.py -q -k rate_limit` and the endpoint module with `-k "blank or whitespace or stripped"`.

Expected: FAIL — the blank question returns 200 with a refusal, and `RATE_LIMIT_PER_MINUTE=0` loads without raising.

- [ ] **Step 3: Implement both**

`src/adapters/primary/http/schemas.py`:

```python
from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_LENGTH = 2000


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    language: Literal["en", "es"]

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """min_length=1 passes "   ". Retrieval would then embed whitespace and
        the refusal gate would score noise as if it were a question."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must contain non-whitespace characters")
        return stripped
```

`src/core/config.py`, inside the existing rate-limit block, right after the `int()` conversion:

```python
    if rate_limit_per_minute <= 0:
        raise ValueError(
            f"RATE_LIMIT_PER_MINUTE must be a positive int, got {rate_limit_per_minute!r}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_core_config.py -q && pytest -q -k "query or schema"`

Expected: PASS.

- [ ] **Step 5: Confirm nothing depended on the old laxity**

Run: `pytest -q`

Expected: PASS. A test that posted `""` or `" "` expecting a refusal must be updated to expect 422 — that is this task's intended behaviour change, not a regression to work around.

---

### Task 6: Replace the `_configured` module-level singleton

**Files:**
- Modify: `src/core/telemetry.py:14-36`
- Test: `tests/test_core_telemetry.py` (create if absent)

**Interfaces:**
- Produces: `configure_tracing` stays idempotent, with idempotence derived from the live OpenTelemetry tracer provider instead of a module-level mutable bool. No signature change beyond Task 4's `otlp_endpoint` keyword.

- [ ] **Step 1: Write the failing tests** — `tests/test_core_telemetry.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from src.core.telemetry import configure_tracing

MODULE = Path(__file__).resolve().parent.parent / "src" / "core" / "telemetry.py"


def test_telemetry_has_no_module_level_mutable_state() -> None:
    """CLAUDE.md's prohibited-patterns rule: no module-level mutable singleton.
    `_configured` was the last survivor of the pre-refactor _model/_CACHE/
    _encoding globals the src/ migration eliminated."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)], "telemetry uses `global`"
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_configured" not in assigned


def test_configure_tracing_is_idempotent_across_apps() -> None:
    configure_tracing(FastAPI())
    first = trace.get_tracer_provider()
    configure_tracing(FastAPI())
    assert trace.get_tracer_provider() is first
    assert isinstance(first, TracerProvider)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_core_telemetry.py -q`

Expected: FAIL on `test_telemetry_has_no_module_level_mutable_state` — `_configured` is assigned at module level and mutated through `global`. `test_configure_tracing_is_idempotent_across_apps` passes already; it is there to pin behaviour that must survive the rewrite.

- [ ] **Step 3: Rewrite the guard**

```python
_SERVICE_NAME = "rag4-api"


def _already_configured() -> bool:
    """Idempotence read off the live provider, not off a module-level bool.
    OpenTelemetry's default is a ProxyTracerProvider; once our own SDK
    TracerProvider carrying our service name is installed, a second call is a
    no-op — and unlike a module flag this stays correct when another entry
    point or an earlier test has already installed one."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        return False
    return provider.resource.attributes.get("service.name") == _SERVICE_NAME


def configure_tracing(app: "FastAPI", *, otlp_endpoint: str | None = None) -> None:
    """Sets up a real TracerProvider so spans created via get_tracer() populate
    JsonFormatter's trace_id field (src/core/logging.py). No endpoint means
    spans are still created but never leave the process — a local no-op, not a
    broken setup."""
    if _already_configured():
        return

    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
```

`app` stays an unused-but-present parameter exactly as today — it is the documented seam for FastAPI instrumentation. Keep whatever suppression style the repo already uses if ruff flags it; do not rename it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_core_telemetry.py -q`

Expected: PASS.

- [ ] **Step 5: Confirm trace ids still reach the logs**

Run: `pytest -q -k "logging or telemetry or trace"`

Expected: PASS — `JsonFormatter`'s `trace_id` is fed by `get_tracer()` spans, so a provider regression surfaces here rather than in production.

---

### Task 7: Seal the boundaries and sync the documentation

**Files:**
- Modify: `tests/test_import_invariants.py` (guard-the-guard), `CLAUDE.md`, `docs/adr/001-modular-monolith-src-layout.md`, `docs/architecture/` C4 sources
- Test: `tests/test_import_invariants.py`

**Interfaces:**
- Consumes: the invariant added in Task 1 and the `Settings` fields added in Task 4.
- Produces: no runtime surface. This task stops the prose from asserting the pre-remediation shape.

- [ ] **Step 1: Write the guard-the-guard test** — append to `tests/test_import_invariants.py`:

```python
def test_features_import_check_detects_an_injected_violation(tmp_path: Path) -> None:
    """A checker that silently matched nothing would pass
    test_features_layer_never_imports_fastapi_or_primary_adapters forever."""
    offender = tmp_path / "src" / "features" / "query" / "offender.py"
    offender.parent.mkdir(parents=True)
    offender.write_text(
        "from fastapi import APIRouter\nfrom src.adapters.primary.http.deps import get_settings\n",
        encoding="utf-8",
    )

    assert "fastapi" in _imported_top_level_modules(offender)
    assert "src.adapters.primary.http.deps" in _imported_module_paths(offender)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_import_invariants.py::test_features_import_check_detects_an_injected_violation -v`

Expected: PASS on first run. This is a guard-the-guard test over helpers that already exist — a passing first run is the correct outcome here, not a skipped RED step. A failure means the helpers are broken, and that is the bug to fix.

- [ ] **Step 3: Update `CLAUDE.md`**

Four statements are now false and must be corrected in place:

1. Commands section, `INDEX_PROFILE` paragraph — it says the variable "is **not** a `Settings` field (mirrors `expansion_mode`)... read and validated only in `cli.run()`". Replace with: it **is** a `Settings` field (`index_profile`), read and validated in `load_settings()`, still env-overridable, still `contextual-v1` by default; `expansion_mode` remains the deliberately non-`Settings` one.
2. Conventions, "Modular monolith with ports/adapters" bullet — it says `src/adapters/primary/http/` **+ `src/features/query/router.py`** are the only places FastAPI is imported outside `src/main.py`. Drop the router clause; `src/features/` is now FastAPI-free and `tests/test_import_invariants.py` enforces it.
3. Conventions, Config/env-loading bullet — add that `src/core/paths.py` is the single repo-root authority, and that `load_settings()` now also owns `INDEX_PROFILE`, `DEPLOYED_SHA` and `OTEL_EXPORTER_OTLP_ENDPOINT`, and rejects a non-positive `RATE_LIMIT_PER_MINUTE`.
4. Conventions, folder-layout bullet — add `src/core/paths.py` and `src/features/retrieval/chunk_store.py`, and move the router into the `src/adapters/primary/http/` entry.

- [ ] **Step 4: Update the ADR and the C4 diagrams**

Add a dated "2026-09-04 architecture remediation" section to `docs/adr/001-modular-monolith-src-layout.md` recording: router moved to the primary adapter; `load_chunks` split out of the build CLI; `src/core/paths.py` introduced; three env vars folded into `Settings`; `telemetry._configured` removed. Then:

```bash
grep -rn "features/query/router\|features\.query\.router" docs/
```

Every hit in the `docs/architecture/` mermaid sources becomes `adapters/primary/http/routes`.

- [ ] **Step 5: Run the full bucket verification**

```bash
pytest -q
ruff check src tests
mypy src
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.regression_set_integrity --verify
python -m src.features.evaluation.gate_holdout_integrity --verify
```

Expected: all green, with the test count at 547 plus the tests added here. Stop at green — committing needs the owner's separate authorization.
