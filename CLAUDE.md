# CLAUDE.md — Manufacturing Knowledge RAG Assistant

Project-specific instructions for Claude Code sessions working in this repo. Read this AND `SPEC.md` before changing anything, and run `pytest` before starting work to confirm the baseline is green.

## How to start a session in this repo

1. Read `SPEC.md` in full (scope, acceptance criteria, no-goals, data-honesty and language policy, phase status).
2. Read this file in full.
3. Run `pytest` and confirm all tests pass before making changes. If they don't, that's the first thing to fix — don't build on a red baseline.
4. Before adding or changing anything under `corpus/`, read `corpus/SOURCES.md` — every corpus file must have a matching row there.

## Tech Stack

Full stack in use across all three phases:

- Python 3.11+
- `pyyaml` — frontmatter parsing
- `tiktoken` (cl100k_base) — token counting for chunk sizing (NOT the embedding model's tokenizer — see the Language Policy note in `SPEC.md`)
- `pytest` — tests
- `sentence-transformers`, `chromadb`, `rank-bm25`, `numpy` (Phase 1-2) — embeddings, vector store, BM25 lexical index, hybrid retrieval
- `fastapi`, `uvicorn[standard]`, `streamlit`, `groq`, `openai`, `pydantic`, `httpx`, `python-dotenv` (Phase 3) — generation API, UI, LLM clients, config/env loading

## Commands

- `python -m ingestion.run` — run the ingestion pipeline over the full corpus; prints document count, chunk count, public/synthetic breakdown, and total corpus size; writes `ingestion/output/chunks.jsonl` (gitignored, regenerated on every run).
- `python -m retrieval.build_index` — embeds all chunks (`BAAI/bge-m3`) and builds the ChromaDB vector store + BM25 lexical index under `retrieval/output/` (gitignored, regenerated on every run); requires `ingestion/output/chunks.jsonl` to already exist.
- `python -m eval.run_eval` — runs the hand-written eval set through the hybrid retriever, writes a versioned recall@k/MRR report to `eval/reports/`.
- `python -m eval.threshold_analysis` — sweeps the refusal cosine-similarity threshold against the eval set to find the answerable/unanswerable separation point.
- `python -m eval.run_generation_eval` — runs the eval set end-to-end through retrieval + generation, checking citation correctness and refusal behavior.
- `uvicorn api.main:app --reload` — run the FastAPI backend locally (serves `GET /health`, `POST /query`).
- `streamlit run ui/streamlit_app.py` — run the UI locally; expects the backend already running, reads the `API_BASE_URL` env var (default `http://localhost:8000`).
- `docker build -t rag4 .` / `docker run -p 7860:7860 rag4` — combined-container deploy build/run (FastAPI + Streamlit + nginx in one image). Not verified end-to-end in this repo's dev environment (Docker Desktop's WSL2 backend was unavailable) — review before relying on it for a real deploy.
- `pytest` — run all tests.

## Conventions

- Python: `snake_case` for modules, functions, and variables; `PascalCase` for classes/dataclasses.
- One dataclass per concept in `ingestion/metadata.py` — don't scatter metadata field definitions across multiple files.
- Type hints are required on all function signatures.
- Docstrings: one line, only where the behavior is genuinely non-obvious (a hidden constraint, a subtle invariant, a workaround). Do not write docstrings that just restate the function name in prose. Prefer none over a redundant one.
- No inline comments explaining *what* code does — the code should read clearly enough that it doesn't need one. A comment is justified only for *why*, when the why isn't obvious from context.
- **Mocking convention**: real LLM API calls are the one thing in this repo deliberately kept out of tests, for cost and determinism. `api/llm_client.py`'s own tests patch the provider SDK clients (`api.llm_client.groq.Groq`, `api.llm_client.openai.OpenAI`) directly; higher-layer tests (`api/generation.py`, `api/main.py`'s endpoints, `eval/run_generation_eval.py`) that would otherwise transitively trigger a real LLM call instead patch the module boundary (`api.llm_client.generate_structured`, or `api.generation.answer_question` when the LLM call isn't the thing under test), sometimes alongside `retrieval.hybrid.retrieve`/`retrieval.vector_store.get_collection` to keep an endpoint- or eval-script-level test fast and deterministic. This is a scoped exception, not a new general testing policy — ingestion, retrieval, and eval-metrics tests are all real integration tests against real models/indexes, deliberately, and a future module should default to that same style unless it specifically needs to avoid live API spend.
- **Config/env-loading pattern**: `api/config.py`'s `Settings` frozen dataclass + `load_settings()` function is the one place environment variables are read. `.env` (gitignored) is loaded via `python-dotenv` only if present — never required, since HF Spaces injects secrets as real env vars at runtime, not a `.env` file. Config-typo values (e.g. an invalid `LLM_PROVIDER`, a non-float `REFUSAL_COSINE_THRESHOLD`) raise `ValueError` at load time; genuinely-optional values (missing API keys) default to `None` without raising and are only checked when actually needed. This is the repo's applied instance of the "no silent fallback" rule under Prohibited Patterns below.
- **Structured logging**: `api/logging_setup.py`'s `JsonFormatter` + `configure(level)` writes one JSON object per log line to stdout, which is what HF Spaces captures as container logs; any caller-supplied `extra={...}` fields are passed through generically. There is no external log aggregation — this is intentionally simple, appropriate for a portfolio-scale app, not a production system. Don't over-engineer this later.
- **Citation-integrity design (hard rule)**: citations in API responses are always re-derived from real retrieved-chunk metadata by `chunk_id`, never trusted from LLM-generated citation fields beyond the `chunk_id` itself — see the Prohibited Patterns entry below. `api/generation.py` looks up `document_id`, `document_title`, `section_heading`, and `revision` from the retrieved chunk matching the LLM-supplied `chunk_id`, dropping (logging, not raising) any citation whose `chunk_id` isn't among the retrieved set.
- **LLM/generation module boundary**: `api/` (FastAPI backend + LLM orchestration) and `ui/` (Streamlit) are separate processes communicating over HTTP only. `ui/streamlit_app.py` must never import from `api.generation`, `api.llm_client`, `api.refusal`, or `api.prompts` (the logic/orchestration/I/O modules) directly, even though both live in the same container/repo — this matches the deployed architecture (nginx proxies to two independent processes) and keeps the API genuinely independently testable and reachable. `api.messages` is a deliberate exception: it's a data-only bilingual string catalog with no logic and no I/O, and `ui/streamlit_app.py` correctly imports `UI_LABELS` from it directly so both surfaces stay in sync from one source.
- Folder layout: `corpus/{public,synthetic}/`, `ingestion/`, `tests/`, `api/` (`config.py`, `logging_setup.py`, `messages.py`, `prompts.py`, `refusal.py`, `llm_client.py`, `generation.py`, `schemas.py`, `main.py` — FastAPI backend + LLM orchestration), `ui/` (`streamlit_app.py` — Streamlit frontend), `docs/` (placeholder). Don't invent new top-level folders without a reason tied to a specific phase's scope.

## CFR Numbering → Markdown Heading Convention

Federal regulation text (21 CFR Part 211, 29 CFR 1910.1200) is numbered (`§ 211.22`, paragraph `(a)`, sub-paragraph `(1)`), not naturally headed the way an SOP or manual is. When converting CFR text to the corpus Markdown format, each numbered level maps one-to-one to a Markdown heading level:

- `##` — a Subpart (e.g. `## Subpart B—Organization and Personnel`) or, where a section stands with its own numbered paragraphs, the section itself (e.g. `## (a) Purpose`).
- `###` — an individual section (`### § 211.22 Responsibilities of quality control unit`) or a lettered paragraph under a section-level `##`.
- `####` — a numbered sub-paragraph, if it's substantial enough to need its own heading (short numbered items are usually left as plain list items within their parent heading instead — use judgment; the goal is a `section_heading` breadcrumb from the chunker that's still a useful citation, not a heading for every single `(a)(1)(i)`).

Apply this convention consistently if more regulatory text is added to the corpus later — don't invent a different scheme per document.

## Prohibited Patterns

- **No hardcoded API keys.** `.env.example` is tracked with empty placeholder keys; the real `.env` is gitignored. This applies even though no LLM calls happen until Phase 3 — the placeholder exists now so Phase 3 doesn't retrofit it.
- **No chunk emitted with incomplete metadata.** Every chunk must have a non-empty `document_id`, `document_title`, `section_heading`, `source_type`, `source_url_or_note`, `chunk_id`, and `md_line_range`. `source_page_range` is the one documented exception — `null` is valid and expected for every synthetic document (see `ingestion/metadata.py`). If a required field would be empty, the pipeline must raise, not silently drop or default it — this mirrors the citation-mandatory, no-silent-fallback-to-hallucination standard the whole project is built around.
- **No copyrighted PDFs committed to this repo**, ever. See `SPEC.md`'s Data-Honesty Policy.
- **No new corpus file without a matching `corpus/SOURCES.md` row.** `tests/test_corpus_manifest.py` enforces this — don't skip it or mark it `xfail` to get a file in.
- **No silent fallback that could look like hallucination-adjacent behavior** — even in this ingestion-only phase, don't build patterns (e.g. defaulting a missing field to `"unknown"` instead of raising) that a later phase might copy and use to mask a real gap.
- **No citation field trusted from LLM output beyond `chunk_id`.** `api/generation.py`'s citation resolution always looks up `document_id`/`document_title`/`section_heading`/`revision` from the real retrieved chunk's metadata, never from the LLM's own JSON output — this is what prevents citation hallucination; a change that starts trusting LLM-supplied citation text for these fields would reintroduce exactly the failure mode this project is built to avoid.

## Scope Discipline

This is a phased project — see `SPEC.md` for what's in scope per phase. Do not add embeddings, a vector store, retrieval logic, LLM calls, FastAPI endpoints, a Streamlit UI, an evaluation set, or deployment/CI config during Phase 1, even if it looks like small, easy additions. Wait for the phase that scopes that work.
