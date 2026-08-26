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
- **Forward constraint on Phase 2's embedding model** (recorded now so Phase 2 doesn't inherit a trap from Phase 1's chunking decision): the embedding model MUST be multilingual AND have `max_seq_length` — as counted by the model's own tokenizer, not by the `tiktoken` cl100k_base counter used for Phase 1 chunk sizing — greater than or equal to the chunk upper bound (600 tokens). A model such as `paraphrase-multilingual-mpnet-base-v2` (max_seq_length 128) is **disqualified**: it would silently truncate roughly 75% of a 600-token chunk and produce meaningless retrieval.
  - **Correction recorded during Phase 2 (this note originally suggested `intfloat/multilingual-e5-base` as a viable candidate — it is not):** the real corpus's max chunk size turned out to be 699 tokens (tiktoken), higher than the 600-token figure this note reasoned from, and `multilingual-e5-base`'s 512-token window falls short of even the original 600 bound. Phase 2 selected `BAAI/bge-m3` instead (max_seq_length 8192, natively multilingual, no `trust_remote_code` needed) and verified it against every real chunk's token count under the model's own tokenizer before indexing — see Phase 2 Status below.
- **Forward constraint on Phase 3's generation prompt and UI**: the generation prompt must instruct the model to answer in the question's language while keeping citations in English (matching the corpus language), and the UI's language toggle affects UI chrome and default query-language assumption, not corpus content.

## Budget & Stack

- **Budget**: $0-20/month for the whole project.
- **Stack**: Python, FastAPI (Phase 3), ChromaDB or FAISS (Phase 2), Sentence-Transformers (Phase 2), Streamlit (Phase 3) — all open-source/free.
- **LLM generation** (Phase 3, out of scope this phase): Groq API (free tier, Llama 3.x) as primary, OpenAI gpt-4o-mini as a low-cost fallback. Explicitly **not** a locally-hosted LLM — the deployment target (Hugging Face Spaces, Docker) cannot reliably serve local LLM inference within budget.

## Deployment Target

**Oracle Cloud "Always Free" Ampere A1 (ARM/aarch64) VM, Docker.** Changed from the original Hugging Face Spaces target — see the 2026-08-25 pivot note in Phase 3 Status below for why and what this changes architecturally (amd64 → aarch64).

**Superseded, original Phase 3 target — kept as historical record:** Hugging Face Spaces, Docker SDK. The Phase 2-planning flag below (which predicted exactly this outcome) turned out to be correct: re-verified live at actual Phase 3 completion, and HF's free tier no longer permits *running* a Docker SDK Space at all, not just creating one — PRO (~$9/mo) is required with no free workaround, confirmed via HF's own docs/forum reports as a mid-2026 policy change. The project owner chose not to pay for PRO; see Phase 3 Status for the resulting pivot.

**Flag recorded during Phase 2 planning (confirmed accurate at Phase 3 completion — see above):** HF's current docs and community forum threads (checked live, 2026) indicate that *creating* a Docker-SDK Space now requires a paid PRO plan (~$9/mo) for personal accounts — free CPU Basic hardware is still free once a Space exists, but a free account can no longer create a new Docker Space to run on it. This still fits the project's $0-20/month budget, but the deploy step is not literally $0 as this document's earlier phrasing implies. HF's policies here have been in flux, so re-verify rather than trust this note by the time Phase 3 starts.

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

## Phase 2 Status

**Status: Complete.**

Built: `retrieval/` (`embedder.py`, `vector_store.py`, `bm25_index.py`, `hybrid.py`, `build_index.py`) and `eval/` (`eval_set.json`, `hash_eval_set.py`, `metrics.py`, `run_eval.py`) with tests (`tests/test_embedding_model_fits_corpus.py`, `test_hybrid_retriever.py`, `test_eval_metrics.py`, `test_eval_set_integrity.py`).

**Embedding model**: `BAAI/bge-m3` (multilingual, `max_seq_length` 8192 under its own tokenizer — see the Language Policy correction above). **Deploy-target memory footprint, checked before committing to this model (explicit decision, not left implicit):** this document's Deployment Target (below) is Hugging Face Spaces, Docker SDK. Its free CPU Basic hardware is 2 vCPU / 16GB RAM (verified live against HF's current Spaces docs during Phase 2 planning, not assumed from training data). `bge-m3`'s ~2.27GB weights plus the sub-1MB Chroma/BM25 indexes for 228 chunks plus a Streamlit+FastAPI process land well under 16GB with comfortable margin — the tighter ~1GB-RAM constraint that motivated this check applies to Streamlit Community Cloud, which is not this project's deploy target. No chunker changes or lighter-model fallback were needed. **Vector store**: ChromaDB (`PersistentClient`, cosine distance), chosen over FAISS because it stores full chunk metadata alongside each vector, so retrieval results carry complete citation info with no separate id→metadata sidecar. **Lexical index**: `rank_bm25` (BM25Okapi) over the same 228 chunks. **Fusion**: Reciprocal Rank Fusion, k=60, combining top-20 semantic and top-20 BM25 results per query — chosen over a weighted sum because RRF needs no score normalization between BM25's unbounded scores and cosine similarity's bounded range, and needs no weight to justify.

**Scope clarification vs. this document's original Phase 2 acceptance criterion "Threshold-based refusal is implemented":** Phase 2 computes and reports a fused confidence score for every query, including the unanswerable eval subset, but does **not** implement the actual gating/refusal decision. That threshold-picking and refusal behavior is deferred to Phase 3, which will consume Phase 2's score distributions (below) to pick a real threshold. This was an explicit scope decision made and flagged before implementation, not an oversight.

**Evaluation results** (`python -m eval.run_eval`, eval_set v1.0.0, git commit `eab132866efc18bfd40ec6c61309690bbf2d757e`):

- **Recall@3**: 0.700
- **Recall@5**: 0.833 (clears the ≥0.7 bar — no iteration on chunking/fusion was needed)
- **MRR**: 0.621
- **Recall@5 by language**: English (n=23) 0.913; Spanish (n=7) 0.571 — the multilingual embedding model works cross-lingually. **Caveat on the Spanish number's precision:** with only 7 questions, each individual hit/miss swings recall@5 by ~14 percentage points — one different outcome moves 0.571 to 0.714. This sample is too small to distinguish "the multilingual model is genuinely weaker on Spanish" from "this particular set of 7 questions happened to be harder." Do not treat 0.571 as a stable measurement, and do not use it alone to justify a model change, until a later phase enlarges the Spanish eval subset. Not a blocker for Phase 2 — the brief asked for 6-8 Spanish questions and that was met — but the number should be read with this uncertainty attached.
- All 5 complete misses (q002, q014, q017, q018, q026) were checked by hand: each retrieved a real, topically-adjacent chunk from the *correct* document (e.g. q002 pulled the "what is lockout/tagout" FAQ chunk instead of the "Commonly Used Terms" glossary chunk that defines "Lockout" specifically) — legitimate near-miss retrieval difficulty from overlapping topical content, not a fusion or indexing bug.

**Important finding for Phase 3, surfaced honestly rather than glossed over:** the unanswerable subset's top-1 RRF fused scores (0.0278–0.0328) overlap heavily with answerable-question fused scores, and 3 of the 10 unanswerable questions hit the mathematical *maximum* possible fused score (0.0328 — rank 1 in both the semantic and BM25 lists) despite having no genuinely relevant chunk in the corpus. This is a structural property of RRF: it scores rank position, not relevance magnitude, so a query with no good match still gets a "confident-looking" fused score from whichever chunk happens to rank best in both lists. **Conclusion: Phase 3's refusal threshold should be built on raw semantic (cosine) similarity or a calibrated confidence signal, not the RRF fused score.** The full per-question score distribution is in `eval/reports/retrieval_report_v1.0.0.md`.

**Bilingual eval design**: 7 of the 30 answerable questions are posed in Spanish against the English-only corpus (cavitation/NPSH, valve types, CGMP expiration dating, hazard communication SDS sections, carbon monoxide IDLH, and LOTO/machine-guarding cross-reference) — added so the "multilingual" requirement has real evidence behind it this phase rather than resting on the model card alone.

**No LLM calls exist anywhere in Phase 2's code** — verified by grep across `retrieval/` and `eval/` for common LLM client imports and API call patterns; none found.

**Addendum, 2026-08-24 — non-deterministic MRR bug found and fixed before Phase 3 work began.** Two runs of `python -m eval.run_eval` reported different MRR values (0.621 vs. a reproduced 0.604) even though Recall@3 and Recall@5 stayed stable across both runs. Root cause: `retrieval/hybrid.py`'s RRF fusion sort (`fused.sort(key=lambda result: result.fused_score, reverse=True)`) had no deterministic tie-break. The fused-results list is built by iterating `set(semantic_by_id) | set(bm25_by_id)` — a Python `set`, whose iteration order is hash-randomized per process (`PYTHONHASHSEED` defaults to random) — so whenever two or more chunks landed on the same `fused_score` (common with RRF, since it scores rank position rather than a continuous relevance signal), Python's stable sort preserved whatever set-iteration order happened to occur that run, silently reordering tied results and shifting which chunk landed at rank 1 for a reciprocal-rank calculation. Fixed by adding `chunk_id` as a deterministic secondary sort key: `fused.sort(key=lambda result: (-result.fused_score, result.chunk_id))`, refactored into an importable `_sort_fused_results()` helper in `retrieval/hybrid.py` so a hand-built-fixture regression test (`tests/test_hybrid_retriever.py::test_fused_score_ties_break_deterministically_on_chunk_id`) can exercise the real tie-break logic without requiring the built index. After the fix, two consecutive `run_eval` runs produced byte-identical reports with **MRR: 0.637** (the value differs from both prior readings of 0.621 and 0.604 because ties are now broken consistently rather than randomly — this is the correct, reproducible number going forward, not a regression).

## Phase 3 Status

**Status: Implemented, not yet deployed.** `api/` (config, structured logging, bilingual message catalog, refusal threshold gate, EN/ES prompts, Groq/OpenAI LLM client with JSON repair and provider fallback, generation orchestration, FastAPI app) and `ui/streamlit_app.py` are built and tested. Local Docker verification is complete (real `docker build`/`docker run`, `/health` and nginx-routing checks all passing — see below); live deployment has not happened yet.

**Deployment-target pivot, 2026-08-25 — Hugging Face Spaces dropped as primary deploy target; Oracle Cloud "Always Free" Ampere A1 (ARM) VM chosen instead.** At Phase 3 completion, re-verifying the Phase 2-planning flag above confirmed it: HF's free tier no longer permits *running* a Docker SDK Space at all (not just creating one) — PRO (~$9/month) is required, with no free workaround, and static Spaces cannot run this project's container. The project owner decided not to pay for HF PRO. Given the project's $0-20/month budget preference, Phase 3 deployment pivots to Oracle Cloud's "Always Free" Ampere A1 ARM VM as the primary $0, always-on host — a real architecture change (amd64 → aarch64 target), requiring ARM/aarch64 wheel-compatibility verification of every native-compiled dependency (torch, tokenizers, safetensors, onnxruntime, chromadb's vector-index bindings, pyarrow, grpcio, etc.) before deployment can proceed; see the compatibility check below. Google Cloud Run is flagged as a backup managed-hosting option if the Oracle ARM path proves intractable (capacity limits, or an unresolvable ARM wheel gap). A free Hugging Face Static Space (no PRO required) or Hub model/dataset card remains planned as a recruiter-visibility showcase page linking to the live Oracle-hosted demo — this doesn't touch compute and is unaffected by the PRO paywall.

**Note, 2026-08-25 — actual generation model in use is `openai/gpt-oss-120b` via Groq, not a Llama 3.x model as originally specified. Approved as a formal deviation from the Phase 3 prompt's "Groq API (Llama 3.x family)" specification (project owner sign-off, 2026-08-25).** `GROQ_MODEL` was changed from `llama-3.3-70b-versatile` (retired by Groq, confirmed via `GET https://api.groq.com/openai/v1/models`) to `openai/gpt-oss-120b` in commit `b73e69d`. This is a model-family deviation, not a same-family swap — Groq's active catalog currently has no viable Llama 3.x chat model at all (the only remaining Llama-family entries are small prompt-injection-classifier models, not usable for generation), and moving primary generation off Groq entirely to preserve the Llama family would be a larger architectural change with its own cost and complexity implications, not justified by this deviation alone. Accepting a different model family on the existing primary provider was judged the smaller, cheaper change. **This deviation does not change the primary/fallback logic**: `generate_structured()` in `api/llm_client.py` selects between providers (`groq`/`openai`), not specific model names, so the fallback path to OpenAI `gpt-4o-mini` on a Groq failure is unaffected — only the model called *within* the Groq branch changed.

**Superseded by the 2026-08-25 addendums below** (the threshold override to 0.5999, and the real `run_generation_eval` results showing correct-refusal rate 0.900 overall) — the paragraph immediately below is kept only as the historical record of the reasoning that led to the threshold override, not as current status. Its one still-relevant point, that the headline correct-refusal-rate metric is measured as gate-refusal-OR-LLM-self-refusal rather than the gate alone, remains true and is carried forward into the addendums below.

**Honest disclosure, 2026-08-24 — the calibrated refusal threshold alone does not clear the plan's headline correct-refusal-rate target.** At the calibrated threshold (`REFUSAL_COSINE_THRESHOLD = 0.5599`, chosen by `eval/threshold_analysis.py`), the sweep table in `eval/reports/threshold_analysis_v1.0.0.md` shows only **5 of 10** unanswerable eval questions are correctly refused by the threshold gate alone — well short of the plan's stated correct-refusal-rate target of **≥ 0.80**. No single threshold value anywhere in the sweep clears both the ≥0.80 correct-refusal target and the ≤0.10 false-refusal cap simultaneously on this 40-question eval set (see the full sweep table in that report).

This is not necessarily a failure of the system, but it is a real, worth-disclosing tension with the plan's original intent (Step 2 was specifically about making refusal a *calibrated, deterministic gate*). `correct_refusal_rate` as actually computed in `eval/run_generation_eval.py` counts a question as correctly refused if *either* the threshold gate refuses it *or* the LLM self-refuses after seeing the retrieved context (`refused: true` in its structured JSON output) — so the measured end-to-end rate could still clear 0.80 if the LLM self-refuses on enough of the remaining 5 unanswerable questions the gate lets through. That means the headline differentiator rests partly on LLM behavior at generation time, not purely on the calibrated gate, and nobody had put these two numbers side by side before this fix wave.

**This will be confirmed or refuted by an actual live `python -m eval.run_generation_eval` run, which has not yet been performed** (it requires real Groq/OpenAI API keys and ~13-15 minutes of wall-clock time — see README). Until that run happens, treat the ≥0.80 correct-refusal-rate target as unverified, not met.

**Addendum, 2026-08-25 — `REFUSAL_COSINE_THRESHOLD` changed from 0.5599 to 0.5999, and the root cause of the tension above.** The paragraphs above were written assuming 0.5599 (the value `eval/threshold_analysis.py`'s `select_threshold()` actually selected) was the only reasonable candidate. On inspection, `eval/threshold_analysis.py` selected 0.5599 *correctly*, exactly as the plan's Step 2 rule specifies — it is not a bug. The real issue is that **Step 2's selection objective and Step 6's downstream acceptance targets are two independently-defined formulas that were never reconciled, and they disagree here**:

- Step 2's objective is `unanswerable_correctly_refused − answerable_wrongly_refused`, an unweighted 1-for-1 trade, with ties broken toward the *lower* threshold. Per the sweep table in `eval/reports/threshold_analysis_v1.0.0.md`, 0.5599 (5 correct, 0 wrong → objective 5) and 0.5999 (7 correct, 2 wrong → objective 5) are **exactly tied** on this objective, so the tie-break rule picked 0.5599, the lower of the two — the code did exactly what the rule says.
- Step 6's acceptance targets are two independent caps, not a linear trade: correct-refusal rate ≥ 0.80 (8/10) and false-refusal rate ≤ 0.10 (≤3/30). Judged against *those* targets specifically, 0.5999 (7/10 = 0.70 correct, 2/30 ≈ 0.067 wrong, comfortably inside the cap) is materially closer to clearing both than 0.5599 (5/10 = 0.50 correct, 0/30 wrong) — Step 2's objective was simply blind to Step 6's targets when it broke the tie.

Given this, `REFUSAL_COSINE_THRESHOLD` is now **0.5999** (updated in `.env.example` and `api/config.py`'s default), a deliberate override of `select_threshold()`'s literal output, not a rerun with different data. `eval/threshold_analysis.py` and its generated report are left unchanged as the accurate historical record of what the approved Step 2 rule actually produces — the override is documented here, not hidden by silently regenerating the report with different logic. At 0.5999, the threshold gate alone now correctly refuses **7 of 10** unanswerable questions (0.70), still short of the ≥0.80 target on its own, but closer than 0.5599's 0.50 — the live `run_generation_eval` run is still needed to confirm the final end-to-end rate once LLM self-refusal is included.

**Addendum, 2026-08-25 — false-refusal rate (0.200) exceeds the plan's ≤0.10 cap; accepted as a documented exception, not a passing result.**

The live `run_generation_eval` run (with `REFUSAL_COSINE_THRESHOLD = 0.5999` and the retired Groq model replaced — see the commit history) measured false-refusal rate = 0.200 (6/30), above the plan's ≤0.10 target. Breaking down the 6 cases: 2 (q009, q024) were refused by the confidence gate alone. The remaining 4 (q014, q017, q018, q026) passed the gate but were self-refused by the LLM after retrieval failed to surface the expected chunk (`retrieval_succeeded = False` for all 4) — the LLM correctly declined rather than answering from unsupported context, exactly as the citation-integrity design requires. All 4 fall in the eval set's 7-question Spanish subset, already flagged in Phase 2's status as too small (n=7) to be a stable measurement.

`retrieval/bm25_index.py`'s tokenizer (`\w+` regex, lowercased, no stemming/stopword removal/accent normalization — read directly, not re-run) is fully language-agnostic, applying identical treatment to English and Spanish text; since the corpus itself is English-only, this means BM25 contributes essentially no lexical signal for Spanish queries, leaving Spanish retrieval to rest almost entirely on `bge-m3`'s cross-lingual semantic embeddings rather than benefiting from hybrid fusion the way English queries do — plausible context for why Spanish retrieval underperforms, though no action is being taken on it this phase.

This is accepted as a known limitation rather than a passing metric: the ≤0.10 cap is not met on its own terms, and no reinterpretation of the denominator changes that. Correct-refusal rate (0.900) does clear its ≥0.80 target. **No further action scheduled this phase — flagged for a future iteration once the Spanish eval subset is enlarged beyond n=7.**

**Addendum, 2026-08-25 — manual citation-accuracy and faithfulness review: 14/30 (0.467) on both metrics, short of the ≥0.90 target — but this reflects partial review coverage, not a quality failure.** Per the plan's decision to grade these two headline metrics by human review rather than LLM-as-judge, the project owner manually reviewed `eval/reports/manual_review_checklist_v1.0.0.csv`'s 30 answerable-subset rows and recorded `citation_accuracy_pass`/`faithfulness_pass` verdicts for 14 of them (q001, q003, q004, q006, q008, q010, q011, q012, q013, q015, q019, q022, q023, q030). **All 14 reviewed rows passed both metrics — 14/14 (1.000).** The remaining 16 rows are still blank (not yet reviewed), not failing verdicts.

- **citation_accuracy_pass: 14/30 = 0.467** — below the ≥0.90 (≥27/30) target as measured across all 30 rows, but this is an artifact of 16 ungraded rows counting as non-passing in a strict True/30 count, not 16 documented failures.
- **faithfulness_pass: 14/30 = 0.467** — same situation, same caveat.

Two of the 14 reviewed rows needed judgment calls, both recorded in `reviewer_notes` and approved: q012 (a likely typo, "líquidos incomprensibles" for "líquidos incompresibles", judged not to affect factual content) and q023 (the citation set includes an extra Definitions section entry alongside the correct Section 4.3 citation, judged as an acceptable superset rather than a wrong citation).

**Status of these two metrics: not yet verified against the ≥0.90 target — review is incomplete, not failing.** Completing the remaining 16 rows' manual review is required before citation-accuracy and faithfulness can be reported as passing, failing, or a documented exception; treat both as open until that review finishes.
