# ADR-001: Modular monolith, `src/` layout with vertical slices

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
