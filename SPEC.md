# SPEC — Manufacturing Knowledge RAG Assistant

This is Portfolio Project 4: a bilingual (EN/ES), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality procedures, with threshold-based refusal when the retrieved context doesn't support a confident answer.

It follows PARO (production FastAPI/PostgreSQL backend, OEE/downtime domain modeling, Decimal-over-float discipline), QualityOps (SPC/Cp-Cpk statistical rigor, cross-validated against Minitab with SHA-256 hashes), and the DMAIC case study (honest synthetic/projected data disclosure) in the same portfolio. This project's differentiator is applied AI on industrial data with real, inspectable evidence — not a generic "chat with your PDF" demo — plus a genuine bilingual use case (English/Spanish industrial workforce, e.g. plants in Northern Mexico).

This document is the single source of truth for scope, acceptance criteria, and policy across all phases. Every phase updates its own "Phase N status" section at the end of this file when complete.

## Whole-Project MVP Scope

- **Phase 1 — Corpus & Ingestion** (this phase): a real+synthetic document corpus with a verified source manifest, and a heading-aware chunking pipeline that produces fully-metadata-complete chunks. No embeddings, retrieval, LLM calls, or UI.
- **Phase 2 — Retrieval**: embed Phase 1's chunks with a multilingual sentence-transformers model, index them in ChromaDB or FAISS, and build a retriever with a held-out evaluation set and threshold-based refusal logic (i.e., the system must be able to say "insufficient context to answer" rather than guessing).
- **Phase 3 — Generation, UI, Deployment**: LLM-backed answer generation (Groq/Llama 3.x primary, OpenAI gpt-4o-mini fallback) with mandatory source citations, a FastAPI backend, a Streamlit UI with an EN/ES toggle, and deployment to Hugging Face Spaces (Docker SDK).

## Whole-Project Acceptance Criteria

**Phase 1 (this phase):**
- `SPEC.md`, `CLAUDE.md`, and `corpus/SOURCES.md` exist and are internally consistent.
- `corpus/SOURCES.md` lists every corpus document with an explicit `public`/`synthetic` label and, for public documents, a real, verified source URL.
- `python -m ingestion.run` processes the full corpus without errors, produces chunks with 100% completeness on all required metadata fields, and prints document count, chunk count, public/synthetic breakdown, and total corpus size.
- `pytest` passes: chunking correctness, metadata completeness, and corpus manifest consistency tests at minimum.
- Git history shows atomic, Conventional-Commits-formatted commits.

**Phase 2 (not started):**
- An embedding model is selected that is multilingual AND has `max_seq_length` (in its own tokenizer) ≥ the chunk upper bound (600 tokens per this phase's chunking — see the Language Policy section below for why this is a hard constraint, not a preference).
- All Phase 1 chunks are embedded and indexed in ChromaDB or FAISS.
- A held-out evaluation set (questions with known-correct source chunks) exists and is used to measure retrieval precision/recall.
- Threshold-based refusal is implemented: below a defined similarity/confidence threshold, the retriever signals "no confident answer" rather than returning weak matches.

**Phase 3 (not started):**
- LLM generation (Groq primary, OpenAI gpt-4o-mini fallback) produces answers that always cite the source document/section for every factual claim.
- The system refuses to answer (rather than hallucinating) when Phase 2's retrieval confidence is below threshold.
- The Streamlit UI offers an EN/ES toggle; questions may be asked in either language; answers are generated in the question's language; citations always reference the original English document/section.
- The system is deployed and reachable on Hugging Face Spaces (Docker SDK), within the $0-20/month budget.

## No-Goals (Whole Project)

These are explicitly out of scope for the MVP — future/buffer-week ideas only, not to be implemented without a separate decision to expand scope:

- No multi-tenant auth system.
- No fine-tuning of any model.
- No Graph-RAG or knowledge-graph retrieval.
- No enterprise SSO.
- No mobile app.

## Data-Honesty Policy

This project's ethical standard, carried over from the DMAIC case study: **never present synthetic data as real.**

- Every corpus document is labeled `public` or `synthetic` in its own frontmatter (`source_type` field) AND in `corpus/SOURCES.md`. The two must always agree — `tests/test_corpus_manifest.py` checks this.
- `public` documents are real U.S. federal government works (OSHA publications, DOE Fundamentals Handbooks, NIOSH guidance, CFR regulatory text) — all public domain under 17 U.S.C. §105, with a real, verified source URL recorded in `SOURCES.md`.
- `synthetic` documents are original works authored for this project to fill gaps public sources don't cover (e.g., a specific fictional machine model, internal-style QMS forms, sample CMMS records). Every synthetic document carries a visible banner at the top of the file stating it is fictional and not a real facility record.
- **No copyrighted material is ever committed to this repository.** If a copyrighted PDF is ever used as a reference in a later phase, only its URL is recorded in `SOURCES.md` (never the file itself), and the file path is added to `.gitignore`'s reserved `corpus/copyrighted/` entry. As of Phase 1, this corpus needed no copyrighted sources — all equipment-manual content came from public-domain DOE Fundamentals Handbooks and OSHA/NIOSH guidance instead of copyrighted manufacturer manuals.
- Every public source URL is verified to actually resolve before a phase is marked done. Where a document is an excerpt of a much larger source (the DOE handbooks, the NIOSH guide), that is disclosed in both the file's banner and `SOURCES.md` — never presented as the complete document.

## Language Policy

- Corpus documents, code, comments, and commit messages are in English.
- The assistant is bilingual (English/Spanish): the Phase 3 UI will offer an EN/ES toggle, and users may ask questions in either language against the English corpus.
- Answers are generated in the language of the question; citations always reference the original English document/section (never translated).
- **Forward constraint on Phase 2's embedding model** (recorded now so Phase 2 doesn't inherit a trap from Phase 1's chunking decision): the embedding model MUST be multilingual AND have `max_seq_length` — as counted by the model's own tokenizer, not by the `tiktoken` cl100k_base counter used for Phase 1 chunk sizing — greater than or equal to the chunk upper bound (600 tokens). A model such as `paraphrase-multilingual-mpnet-base-v2` (max_seq_length 128) is **disqualified**: it would silently truncate roughly 75% of a 600-token chunk and produce meaningless retrieval. A model such as `intfloat/multilingual-e5-base` (512-token window) is a viable starting candidate, but Phase 2 must verify the actual `max_seq_length` of whichever model is chosen against real corpus chunks before indexing anything.
- **Forward constraint on Phase 3's generation prompt and UI**: the generation prompt must instruct the model to answer in the question's language while keeping citations in English (matching the corpus language), and the UI's language toggle affects UI chrome and default query-language assumption, not corpus content.

## Budget & Stack

- **Budget**: $0-20/month for the whole project.
- **Stack**: Python, FastAPI (Phase 3), ChromaDB or FAISS (Phase 2), Sentence-Transformers (Phase 2), Streamlit (Phase 3) — all open-source/free.
- **LLM generation** (Phase 3, out of scope this phase): Groq API (free tier, Llama 3.x) as primary, OpenAI gpt-4o-mini as a low-cost fallback. Explicitly **not** a locally-hosted LLM — the deployment target (Hugging Face Spaces, Docker) cannot reliably serve local LLM inference within budget.

## Deployment Target

Hugging Face Spaces, Docker SDK. Documented here as the Phase 3 target; no deployment or CI/CD work happens before Phase 3.

## Repository Structure

```
RAG4/
├── SPEC.md                 # this file
├── CLAUDE.md                # project conventions, prohibited patterns, session-start checklist
├── README.md                 # setup + how to run ingestion locally
├── LICENSE                    # MIT (code + synthetic corpus docs; public corpus docs are public domain)
├── requirements.txt
├── .env.example                # empty placeholder keys for Phase 3
├── .gitignore
├── corpus/
│   ├── SOURCES.md              # source manifest — see Data-Honesty Policy above
│   ├── public/                 # 9 real, public-domain documents
│   └── synthetic/               # 5 clearly-labeled fictional documents
├── ingestion/                    # Phase 1: chunking pipeline
│   ├── metadata.py
│   ├── loader.py
│   ├── chunker.py
│   └── run.py                     # entrypoint: python -m ingestion.run
├── tests/
├── api/                          # empty placeholder, Phase 3
└── docs/                         # empty placeholder
```

## Phase 1 Status

**Status: Complete.**

Built: `SPEC.md`, project `CLAUDE.md`, the full repo skeleton, a 14-document corpus (9 real public-domain government documents + 5 clearly-labeled synthetic documents) with a verified `corpus/SOURCES.md` manifest, and a heading-aware ingestion/chunking pipeline (`ingestion/metadata.py`, `loader.py`, `chunker.py`, `run.py`) with tests (`tests/test_chunker.py`, `test_metadata_completeness.py`, `test_corpus_manifest.py`).

All 7 directly-fetchable public source URLs (3 OSHA publications, 3 DOE Fundamentals Handbook PDFs, 1 NIOSH guide) were verified live during ingestion prep. The 2 CFR regulatory sources were retrieved from GPO's `govinfo.gov` bulk XML (the annual CFR edition) rather than the interactive `ecfr.gov` reader, which blocked automated retrieval — this substitution is documented in `corpus/SOURCES.md` and in each affected file's frontmatter.

Verified end-to-end on 2026-08-24: `python -m ingestion.run` processes all 14 documents (9 public, 5 synthetic) into 228 chunks (48,850 words / 323,482 characters of total corpus body text) with zero errors, and `pytest` passes all 15 tests (chunker — including sibling-section merging and a line-range regression test, metadata completeness, corpus manifest consistency). Re-run either command locally to reproduce these numbers — see README for exact commands.

Chunk-size distribution (tokens, tiktoken cl100k_base): mean 291, median 251, min 18, max 699; 27.6% of chunks land in the exact 400-600 target band, and 84.2% are at or above 100 tokens. An initial version of the chunker only sub-split oversized sections and left many short sibling sections (e.g. one-sentence CFR provisions) as separate tiny chunks — caught via a distribution check during phase review (mean chunk size looked low relative to the token target), fixed by merging undersized same-parent sibling sections (see the `fix(ingestion): merge undersized sibling sections` commit).

**Manual spot-check of the fix, and what it found.** Per a second round of review, the smallest and largest post-fix chunks (14 total) were read literally against their source files and cross-checked against `corpus/SOURCES.md`. Every sampled chunk was a complete, grammatically whole idea — none started or ended mid-sentence — and every section heading corresponds to a real, named section of its source document (e.g. real CFR section numbers, real OSHA/DOE/NIOSH subsection titles), not an arbitrary cut. One large chunk turned out to span only half of an 8-row OSHA table (mid-table split); this was confirmed harmless — all 8 rows are present across that chunk and its immediate neighbor, with the split row duplicated in the 15% overlap as designed.

The same spot-check did surface two real defects, both now fixed:
1. **`chunks.jsonl` never persisted actual chunk text** — only metadata pointing back to the source file by line range — contradicting this document's own claim that Phase 2 reads chunks directly with no re-parsing. Fixed by adding a required `chunk_text` field.
2. **`md_line_range` was silently wrong for the last chunk of any multi-chunk section** — `split_into_sections()` trimmed leading/trailing blank lines from a section's text without shifting `start_line`/`end_line` to match, so the line-range metadata under-counted by however many blank lines were trimmed. This didn't corrupt chunk *text* (always assembled by index, never re-sliced from the file) but did corrupt the citation pointer — on the real corpus it made all of 29 CFR 1910.1200(b)'s paragraph (6), a full exemption list, invisible to anything using `md_line_range` to locate a chunk's content. Fixed in `flush()`, with a new regression test asserting the source file's line at `md_line_range`'s start/end matches `chunk_text`'s first/last line, for every chunk in the real corpus.

The remaining small chunks (after the merge fix) are confirmed-by-reading legitimately short standalone content (a section's own preamble before its subsections, a lone short provision with no mergeable sibling under the same parent) rather than fragments — Phase 2's evaluation set is still the right place to confirm this doesn't hurt retrieval quality in practice, but it is not a Phase 1 correctness concern.
