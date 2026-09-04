# ADR-001: Modular monolith, `src/` layout with vertical slices

## Status

**Accepted.** In force. The `src/` layout, the framework-free `src/domain/`, and the
import invariant that enforces it are all still the shipped architecture; the invariant
test was extended (not relaxed) to cover `src/web/`'s HTTP-only boundary as well.

## Context

The app grew phase-by-phase (ingestion → retrieval → generation → API/UI) as
five flat top-level packages: `api/`, `ingestion/`, `retrieval/`, `eval/`,
`ui/`. Nothing enforced a dependency direction between them, the repo wasn't
pip-installable (`pyproject.toml` held only ruff/mypy config), and business
logic (refusal policy, RRF fusion, citation resolution) lived mixed in with
FastAPI/ChromaDB/LLM-SDK code, making it untestable without framework mocks.

## Decision

Adopt a `src/{core,domain,features,adapters,web}` modular monolith:

- `core/` — config, errors, logging, telemetry (cross-cutting, framework-light).
- `domain/` — models, policies (RRF, refusal, citation resolution), ports
  (`Protocol` interfaces) — zero framework/adapter imports, enforced by a
  static AST-based test.
- `features/{ingestion,query,retrieval,evaluation}/` — one vertical slice per
  capability, each owning its own CLI/use-cases/orchestration.
- `adapters/{primary/http, secondary/{embedder,vector,lexical,llm}}/` — the
  only places FastAPI, ChromaDB, the embedding model, and the LLM SDKs are
  imported.
- `web/` — Streamlit, HTTP-only against the API, zero backend imports.

One process, one repo, one API runtime — explicitly not a microservice split
(see ADR-005 for the deploy-shape question this does not settle).

## Consequences

- The import-invariant test becomes possible: `domain/` importing fastapi,
  chromadb, groq, openai, streamlit, or torch is now a build-breaking bug,
  not just a style preference.
- The repo becomes pip-installable (`pip install -e .`), so CLIs run as
  `python -m src.features.<slice>.cli` instead of ad hoc top-level scripts.
- Migration happens in phases (0: freeze, 1: this skeleton, 2: query slice,
  3: ingestion/retrieval-CLI/evaluation, 4a: UI + deploy hygiene, 4b:
  deferred deploy-shape change) so `pytest` stays green throughout — the old
  flat packages and the new `src/` tree coexist until each phase's cutover,
  rather than one large simultaneous rewrite.

## 2026-09-04 architecture remediation

An external architecture audit confirmed six structural leaks in the layout
this ADR established. None changed behaviour; all are now closed, and each is
pinned by a test so it cannot reopen:

- **The query router lived in `src/features/query/router.py`**, so `src/features/`
  imported `fastapi` and reached into `src/adapters/primary/`. Moved to
  `src/adapters/primary/http/routes.py`. `tests/test_import_invariants.py`
  now forbids both, with a guard-the-guard test. Imports of
  `src.adapters.secondary` from `src/features/` stay legal — the eval runners
  construct real adapters deliberately.
- **`app.lifespan` imported `load_chunks` from the index-build CLI**, dragging
  chromadb and sentence-transformers into the serving import path to read chunk
  ids. Split into `src/features/retrieval/chunk_store.py`, which imports only
  stdlib, `src.core.paths` and `src.domain.models`. `cli.py` re-exports both
  names, so the old import path still resolves.
- **Sixteen modules computed their own repo root** with `Path(__file__).resolve()
  .parent.parent.parent[.parent]`, which is off by one the moment a module moves
  between package depths — exactly what the router move does. `src/core/paths.py`
  is now the single authority. It imports nothing but `pathlib`, which is why
  `gate_holdout_integrity.py` can bind `CHUNKS_FILE` from it and still never
  pull in the embedder import chain (the reason it used to redefine the path).
- **`INDEX_PROFILE`, `DEPLOYED_SHA` and `OTEL_EXPORTER_OTLP_ENDPOINT` were read
  outside `load_settings()`**, contradicting this repo's stated one-authority
  config rule. All three are now `Settings` fields. `INDEX_PROFILE` moving in
  reverses an earlier note in CLAUDE.md; the original rationale does not apply,
  because that variable was always meant to be env-overridable (`raw-v1` is the
  documented rollback path). `expansion_mode` remains deliberately outside
  `Settings` so production cannot override it.
- **`src/core/telemetry.py` held a module-level mutable `_configured` flag**, the
  last survivor of the pre-refactor `_model`/`_CACHE`/`_encoding` globals this
  migration eliminated. Idempotence is now derived from the live tracer
  provider, which also stays correct when another entry point installed one.
- **Two input-validation holes**: `question` passed `min_length=1` on `"   "`
  (retrieval would embed whitespace and the refusal gate would score noise —
  now 422), and `RATE_LIMIT_PER_MINUTE=0` loaded fine (the container would boot
  healthy and answer 429 to everyone — now a load-time `ValueError`).

No byte-stable invariant moved: `0.5999`, `0.5500`, RRF `k=60`, `REFUSAL_POLICY`
default `binary`, `expansion_mode` default `off`, served profile `contextual-v1`.
No frozen dataset was re-stamped and no baseline report was overwritten.
