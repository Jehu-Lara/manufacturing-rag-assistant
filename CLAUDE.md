# CLAUDE.md — Manufacturing Knowledge RAG Assistant

Project-specific instructions for Claude Code sessions working in this repo. Read this AND `SPEC.md` before changing anything, and run `pytest` before starting work to confirm the baseline is green.

## How to start a session in this repo

1. Read `SPEC.md` in full (scope, acceptance criteria, no-goals, data-honesty and language policy, phase status).
2. Read this file in full.
3. Run `pytest` and confirm all tests pass before making changes. If they don't, that's the first thing to fix — don't build on a red baseline.
4. Before adding or changing anything under `corpus/`, read `corpus/SOURCES.md` — every corpus file must have a matching row there.

## Tech Stack (current phase)

- Python 3.11+
- `pyyaml` — frontmatter parsing
- `tiktoken` (cl100k_base) — token counting for chunk sizing (NOT the embedding model's tokenizer — see the Language Policy note in `SPEC.md`)
- `pytest` — tests

Future phases add: `sentence-transformers`, `chromadb` or `faiss`, `fastapi`, `streamlit`, an LLM client for Groq/OpenAI. Do not add these dependencies until the phase that needs them.

## Commands

- `python -m ingestion.run` — run the ingestion pipeline over the full corpus; prints document count, chunk count, public/synthetic breakdown, and total corpus size; writes `ingestion/output/chunks.jsonl` (gitignored, regenerated on every run).
- `pytest` — run all tests.

## Conventions

- Python: `snake_case` for modules, functions, and variables; `PascalCase` for classes/dataclasses.
- One dataclass per concept in `ingestion/metadata.py` — don't scatter metadata field definitions across multiple files.
- Type hints are required on all function signatures.
- Docstrings: one line, only where the behavior is genuinely non-obvious (a hidden constraint, a subtle invariant, a workaround). Do not write docstrings that just restate the function name in prose. Prefer none over a redundant one.
- No inline comments explaining *what* code does — the code should read clearly enough that it doesn't need one. A comment is justified only for *why*, when the why isn't obvious from context.
- Folder layout: `corpus/{public,synthetic}/`, `ingestion/`, `tests/`, `api/` (Phase 3 placeholder), `docs/` (placeholder). Don't invent new top-level folders without a reason tied to a specific phase's scope.

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

## Scope Discipline

This is a phased project — see `SPEC.md` for what's in scope per phase. Do not add embeddings, a vector store, retrieval logic, LLM calls, FastAPI endpoints, a Streamlit UI, an evaluation set, or deployment/CI config during Phase 1, even if it looks like small, easy additions. Wait for the phase that scopes that work.
