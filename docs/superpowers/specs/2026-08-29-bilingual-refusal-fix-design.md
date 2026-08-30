# Design — Bilingual / terse-query false-refusal fix

- **Date:** 2026-08-29 (rev. 2026-08-30, review round 1)
- **Status:** Draft for review — round 2
- **Owner:** Jehu-Lara
- **Supersedes / builds on:** the ephemeral "Plan para corregir rechazos bilingües del RAG" (kept as the original hypothesis; this document is the reviewed, evidence-backed version)
- **Scope:** `RAG4` only. Presentation, images, portfolio, and deployment are frozen for the duration of this work.

---

## 1. Problem

The deployed assistant refuses questions it has the source material to answer. Two user-reported live queries, both refused:

| Query | Language | Live result |
|---|---|---|
| `What is the difference between NPSHA and NPSHR?` | en | *Not answered — insufficient information* |
| `¿Cuál es la diferencia entre NPSHA y NPSHR?` | es | *No respondida — información insuficiente* |

The user's broader report: **many questions that should be answerable are refused — some in English, more in Spanish.**

The corpus **does** contain the answer. `corpus/public/doe-hdbk-1018-1-pumps.md` (chunk `doe-hdbk-1018-1-pumps::chunk-0007`, section *Centrifugal Pump Operation > Net Positive Suction Head*) defines NPSHA, NPSHR, both full expansions, and the `NPSHA >= NPSHR` inequality verbatim.

## 2. Evidence (measured locally against the current index, 2026-08-29)

Ran the two failing queries plus variants through the real `HybridRetriever` on the current built index. `REFUSAL_COSINE_THRESHOLD = 0.5999`. Target chunk = `doe-hdbk-1018-1-pumps::chunk-0007`.

| Query | target fused rank | top-1 semantic score | gate decision |
|---|---|---|---|
| `What is the difference between NPSHA and NPSHR?` | **1** | **0.5582** | **REFUSE** |
| `¿Cuál es la diferencia entre NPSHA y NPSHR?` | **1** | **0.5598** | **REFUSE** |
| `...between net positive suction head available (NPSHA) and required (NPSHR)?` | 1 | 0.7215 | answer |
| `¿...altura neta de succión positiva disponible (NPSHA) y la requerida (NPSHR)?` (eval q010) | 1 | 0.6818 | answer |
| `...between net positive suction head available and net positive suction head required?` (expansion, no acronym) | 1 | 0.6468 | answer |
| `What is NPSHR?` | 2 | 0.4988 | REFUSE |
| `¿Qué es la cavitación en una bomba centrífuga y qué la causa?` (eval q009) | 2 (sem rank 1) | 0.5796 | REFUSE |

### What this shows

1. **Retrieval is not the failure for the reported queries.** The correct chunk is fused-rank 1 for both. The refusal gate rejects a correctly-retrieved top result because the *bare-acronym query string* has low cosine similarity to chunk text that spells the term out in full.
2. **Expanding the acronym fixes the reported queries.** Cosine rises 0.558 -> 0.65–0.72 and the gate flips REFUSE -> answer. The expansion carries the signal whether or not the acronym is also present.
3. **The Spanish problem is partly the same gate, not only acronyms.** `q009` (cavitation, no acronym) retrieves the right chunk at semantic rank 1 (0.5796) and is still gate-refused.
4. **Corroborating pattern in `eval/reports/`:** Spanish answerable top-1 cosine clusters 0.58–0.71; English answerable rarely drops below 0.60 (min 0.6028). The threshold was calibrated on a 77 %-English, n=30 set (`threshold_analysis_v1.0.0.md`). `generation_eval_v1.0.0.md` records false-refusal rate 0.200 (6/30), already an accepted-exception failure of the <= 0.10 target, 4 of the 6 Spanish. **This "ES runs ~0.05–0.09 below EN" observation is treated as an open question in §6, not an established fact** — it is answered by the frozen matched-pair data before any per-language threshold is considered.

### Secondary finding (not the reported bug, documented for completeness)

NIOSH "How to Use This Guide / Field Definitions" chunks act as **decoys** for chemical-specific data questions: `q017` (acetone PEL, **English**), `q018` (CO IDLH), `q031` retrieve the front-matter chunks that *define* "PEL"/"IDLH" instead of the chemical's data chunk. This is a ranking/indexing problem, hits English too, and is **out of primary scope** but is a candidate for the contextual-embedding intervention below.

## 3. Root cause

Two independent contributing causes, both real:

- **RC-1 — terse / acronym / cross-lingual queries under-score against verbose chunk text.** The embedded chunk text is the raw body only; a short query using an abbreviation or a non-English surface form lands well below a threshold tuned for verbose English queries, even when it is the rank-1 hit.
- **RC-2 — the refusal threshold and its calibration set are English-biased and never stratified by language.** `threshold_analysis.py` pools all languages; `eval_set` v1.0.0 has only 7 Spanish questions (SPEC.md flags this repeatedly as too small to act on).

## 4. Goals / Non-goals

### Goals
- The two reported NPSHA/NPSHR queries (en + es) return a correct, cited answer.
- Measurable reduction in Spanish false-refusals with **no regression** in English metrics and **no increase** in answering unanswerable questions beyond the documented baseline.
- An enlarged, honest Spanish evaluation basis, **frozen before any intervention is measured**, so decisions rest on data, not n=7.
- Every change measured independently, cheapest intervention first, stop when the frozen acceptance table is cleared.

### Non-goals
- No LLM-based query translation or expansion before retrieval (latency, cost, query exposure to the provider — the original plan's reasoning stands).
- No corpus expansion **unless** the enlarged eval proves a genuine content gap. The DOE handbook already covers NPSH; the reported failure is not a coverage gap.
- No switch of embedding model, vector store, or fusion algorithm.
- No change to the citation-integrity design, the RRF `k=60` / tie-break invariant, or Phase 4b container split.
- No deploy, commit, or push until comparative before/after results are reviewed and explicitly approved.

## 5. Approaches considered

### A. Query-side deterministic term expansion *(original plan)*
Append canonical expansions for a small curated glossary to the query before embedding + BM25; keep the original query for the answer's language.
- **+** Smallest change, no reindex, directly fixes the reported queries (measured).
- **−** Only helps queries containing a known glossary term. Does nothing for `q009`-type Spanish refusals. Risk of overfitting the glossary to the bug currently being debugged.

### B. Index-side contextual chunk embedding
Prepend `document_title > section_heading` to each chunk's *embedding input* at index build time; stored chunk text unchanged.
- **+** Raises similarity broadly, helps the decoy problem and cross-lingual matching, generalises with no per-term curation.
- **−** Requires a full reindex. Won't by itself lift a bare `What is NPSHR?` as far as explicit expansion does.

### C. Recalibrate / restructure the gate
Stratified threshold analysis on the enlarged Spanish set; then a modest global threshold change, a documented per-language threshold, or a two-tier "low-confidence answer" response.
- **+** Addresses RC-2 directly; the only path that helps `q009`-type cases.
- **−** Trades against the "refuse rather than hallucinate" differentiator. Changing `0.5999` touches a documented byte-stable invariant and needs sign-off.

### Recommendation — **A -> B -> C in sequence, measured independently, stop early.**
Do the cheap query-side expansion (A) first — it demonstrably fixes the reported bug. Add contextual embedding (B) only if A alone doesn't clear the frozen table. Only then revisit the gate (C). Corpus expansion (C3) stays deferred.

> **Revision note (2026-08-30, review round 1 — PASS_WITH_FIXES / P1).** Folded in: (i) the eval basis (questions, expected chunks, acceptance criteria) is **frozen in Phase 0 before any intervention is measured**; (ii) EN corpus-attestation and ES translation evidence are **separate tracks** with different rules; (iii) the "unknown acronym still refuses" assertion is dropped — only **passthrough** is testable; (iv) C1 is measured **semantic-only, BM25-only, and both** because expanding either channel moves ranking, not just the gate; (v) C2 states explicitly that the embedding input is prefixed while `documents=` stores the original `chunk_text`; (vi) `eval_set` v1.0.0 and its reports are **immutable** — new output is v1.1.0 with provenance headers; (vii) `regression_eval` output is a **committed, versioned** artifact (decided here); (viii) multi-expansion **order + dedup** tests added; (ix) the `API_KEY` test-isolation fix is a **required precondition**.

## 6. Detailed design

### Acceptance criteria — FROZEN in Phase 0, before any intervention is measured

All pass/fail judgements for C1, C2, and C use **eval_set v1.1.0** and the **regression set**, both frozen (content + expected chunks + this table committed) before the first intervention runs. No question, expected-chunk, or target is added or edited after an intervention has been measured against it. A genuine eval-set defect found mid-flight is fixed as its own reviewed change and **every** intervention is re-measured from scratch.

Per-language, on eval_set v1.1.0:

| Metric | English | Spanish | Rule |
|---|---|---|---|
| Recall@5 (answerable) | >= 0.913 (no regression vs v1.0.0) | >= 0.80 | hard |
| Matched-pair cosine gap (mean EN − ES, same expected chunk) | report | report; target <= 0.05 | evidence, not a gate |
| Correct-refusal rate (unanswerable) | >= 0.90 (no regression vs v1.0.0 0.900) | >= 0.80 | hard |
| False-refusal rate (answerable) | <= 0.10 | material improvement vs ES baseline + honest disclosure if <= 0.10 not met (SPEC precedent) | hard EN / soft ES |
| The 2 reported NPSHA/NPSHR queries (en + es) | correct, cited answer | correct, cited answer | hard |

### Phase 0 — Preconditions & frozen eval basis *(no production behaviour change)*

1. **`API_KEY` test-isolation fix (required precondition).** `test_http_contract_snapshot.py::test_new_app_matches_phase0_snapshot` and `test_http_endpoints.py::test_query_unhandled_exception_returns_500_with_generic_body` must clear/override `API_KEY` the way they already override the rate limiter (e.g. `monkeypatch.delenv("API_KEY", raising=False)` or a `Settings(api_key=None)` dependency override). After this, `pytest` is green in a dev environment that has a local `.env`. Standalone reviewed change, committed before Phase 1. Root cause: §10.
2. **`eval/regression_queries.json`** (data; not under `src/`; own stored SHA-256 via a `regression_set_integrity` check mirroring `eval_set_integrity`, so "frozen" is enforced not just asserted):
   - The 2 reported queries verbatim.
   - en/es **pairs** for: bare acronym, acronym + inline expansion, expansion-only, single-acronym (`What is NPSHR?` / `¿Qué es NPSHR?`), and >= 3 other corpus acronyms from the frozen glossary.
   - Surface variants: `NPSH-A`, `NPSHa`, `net positive suction head`.
   - Control **unanswerables that look on-topic**: e.g. `What is the NPSH margin recommended by API 610?` (API 610 not in corpus) + a Spanish equivalent.
   - Each row: `query`, `language`, `expected_chunk_id` (or `null`), `should_answer` (bool).
3. **`src/features/evaluation/regression_eval.py`**: runs the frozen regression set through the real retriever in **each measured configuration** and writes **committed, versioned** `eval/reports/regression_eval_v1.1.0.md` with a provenance header (git commit, eval_set version+hash, regression_set hash, intervention config, date). One file per eval_set version, regenerated and diffed in review. Committed, not gitignored — decided here.
4. **`eval/eval_set.json` -> v1.1.0**, built and frozen now, before interventions:
   - >= 25 Spanish answerable questions across all corpus domains, including terse/acronym phrasings — not only fully-spelled-out ones.
   - >= 15 Spanish-relevant unanswerable questions (the 10 English unanswerables stay unchanged).
   - One **English counterpart per new Spanish answerable question**, same `expected_chunk_ids`, for the matched-pair gap.
   - `version` -> `1.1.0`; regenerate stored hash with `eval_set_integrity --write`.
   - **v1.0.0 preserved** as `eval/eval_set_v1.0.0.json` (confirm the exact keep-both mechanism with `eval_set_integrity` in review — the current module loads a single `eval_set.json`).
   - CI hash check + `tests/test_evaluation_*` follow `load_eval_set()` to the new version — verify green.
5. **Immutability rule.** `eval/reports/*_v1.0.0.md` are never regenerated or edited. Every new report is `*_v1.1.0.md` with the provenance header from step 3. Comparative tables cite both.

### Phase 1 — Diagnose + measure C1 (query term expansion)

6. **Stratified threshold analysis.** Extend `threshold_analysis.py` to emit per-language and per-(answerable|unanswerable) sweep tables alongside the pooled one. Run against **v1.1.0** -> `threshold_analysis_v1.1.0.md` (v1.0.0 untouched). Measurement only — no threshold change.
7. **Classify** every v1.1.0 miss + every regression failure: gate over-refusal (retrieval OK) / retrieval miss / decoy chunk. Table goes in the companion results doc.
8. **C1 implementation:**
   - **New `src/domain/policies.py :: expand_query(query: str) -> str`** — pure function (matches the existing `fuse_rankings` / `rrf_scores` style). Detects whole-word, case-insensitive glossary keys (`\b` boundaries; `NPSHATEST` and `xNPSHA` must **not** match `NPSHA`). Returns `query + " " + " ".join(expansions)` in a **deterministic order** (glossary insertion order, then expansion-tuple order) and **deduplicated** (an expansion string already present in the query, or shared by two matched keys, appears at most once). No glossary key matched -> returns `query` unchanged (`result == query`).
   - **Glossary `GLOSSARY: dict[str, tuple[str, ...]]`** in the same file. **Two evidence tracks:**
     - *English expansions* — each MUST appear verbatim (case-insensitive) in >= 1 `corpus/` file; one-line comment names the file. Enforced by `test_glossary_english_expansions_are_corpus_attested`.
     - *Spanish renderings* — curated standard technical translations a plant worker would type. **NOT** expected in the English-only corpus. One-line comment per entry with its justification / citation to an authoritative Spanish term source. Enforced only by `test_glossary_spanish_renderings_nonempty_and_distinct` — never by corpus attestation.
     - Dated module docstring records the curation method (all-caps token frequency scan of `corpus/`, then per-term lookup) and the seed set: `NPSHA, NPSHR, NPSH, PEL, IDLH, TWA, REL, LEL, UEL, SDS, PPE, LOTO, CGMP` (final set frozen in steps 2/4).
   - **Call site** — inside `HybridRetriever.retrieve()`. `QueryUseCase` keeps passing the **original** `question` to `build_user_prompt`.
9. **C1 measurement — three configs, each a full run over v1.1.0 + regression set, each its own report section:**
   - **C1-sem**: expanded query -> `vector_store.query` only; BM25 gets the original.
   - **C1-bm25**: expanded query -> `lexical_index.query` only; semantic gets the original.
   - **C1-both**: expanded query -> both.
   Per config and per language: Recall@3/@5, MRR, matched-pair cosine gap, regression-set gate decisions, correct-/false-refusal on v1.1.0. Rationale: expanding a channel changes that channel's **ranking** -> RRF fusion -> the gate's rank-1 pick — not just the gate scalar. We must see which channel(s) help and whether either hurts English.
10. **Decision gate:** a C1 config that clears the frozen acceptance table ships; stop. Else -> Phase 2.

### Phase 2 — C2 (contextual chunk embedding) *(only if no C1 config clears the table)*

11. **Explicit mechanism** in `ChromaVectorStore.build_collection`:
    - `documents = [c.chunk_text for c in chunks]` — **unchanged**. Stored payload, prompt context, and `CitationResolver` input stay the raw chunk text.
    - `embedding_inputs = [f"{c.document_title} > {c.section_heading}\n\n{c.chunk_text}" for c in chunks]` — **new**, passed only to `self._embedder.embed_texts(...)`.
    - `collection.add(ids=ids, embeddings=<from embedding_inputs>, documents=<chunk_text>, metadatas=<unchanged>)`.
    - Net: similarity is computed against heading-prefixed text; retrieval results, citations, and prompt context are byte-identical to today.
    - `assert_fits_max_seq_length(embedding_inputs)` (bge-m3 max_seq_length 8192; prefix < 30 tokens).
12. Requires `python -m src.features.retrieval.cli` reindex. **ADR-008** written and reviewed before C2 lands.
13. **C2 measurement:** same report shape as Phase 1; configs `C2` and (if a C1 config was close) `C1-<best> + C2`. English Recall@5 >= 0.913 is a hard stop.

### Phase 3 — C (gate recalibration) *(only if A+B still don't clear the table)*

14. Threshold decision, from **v1.1.0 stratified data only**, in order:
    a. Single global threshold clears both languages -> keep `0.5999`.
    b. **Documented per-language threshold.** `QueryUseCase` holds `language`; pass it into `RefusalPolicy` (constructor takes `dict[Language, float]`, or the use case selects the value). Requires: SPEC.md Phase 3 status update, `CLAUDE.md` byte-stable-invariants entry update, `.env.example`, ADR, **explicit owner sign-off**.
    c. Two-tier low-confidence response — separate spec, out of scope.

### Phase 4 — C3 (corpus addition) *(only if Phase 1 classification shows a real content gap, not a ranking/gate gap)*

15. Candidate: USDA–NRCS *National Engineering Handbook* Part 623 Ch. 8 (pumping plants). Verify federal authorship — **not contractor-authored** — before use. Full `corpus/SOURCES.md` row + frontmatter + banner. Reindex. Deferred by default; §2 shows NPSH is already covered.

### Phase 5 — Verification gate

16. `pytest` (full, green with `.env` present after Phase 0), `ruff check src tests`, `mypy src`, `eval_set_integrity --verify`, `regression_set_integrity --verify`.
17. All affected `*_v1.1.0.md` reports regenerated; v1.0.0 untouched. SPEC.md Phase 2/3 status updated; `corpus/SOURCES.md` only if Phase 4 ran.
18. Secrets review (`git diff` — no `.env`, keys, PDFs staged).
19. Present comparative before/after (v1.0.0 vs v1.1.0, baseline vs shipped config). **Stop. No commit, push, or deploy without explicit approval** (exact `DEPLOY` confirmation for any deploy).

## 7. Data flow (after C1-both; C1-sem / C1-bm25 route the original string to the other channel)

```
question (original) --+------------------------------------> build_user_prompt  (unchanged - answer language preserved)
                      |
                      +--> expand_query --> expanded query --+--> ChromaVectorStore.query   (semantic)
                                                             +--> Bm25LexicalIndex.query    (lexical)
                                                                   --> RRF fuse --> top-k --> RefusalPolicy gate --> LLM
```

C2 changes only what `build_collection` feeds the embedder; the query path above is identical.

## 8. Testing

| Test | Location | Asserts |
|---|---|---|
| `expand_query` passthrough | `tests/test_domain_policies.py` | no glossary key present -> `result == query` (identical); **no** assertion about downstream refusal |
| `expand_query` known term | `tests/test_domain_policies.py` | `"NPSHA"` present -> each of its expansion strings appended; original query is a prefix of the result |
| `expand_query` case / word boundary | `tests/test_domain_policies.py` | `npsha` matches; `NPSHATEST`, `xNPSHA` do not |
| `expand_query` multi-term order | `tests/test_domain_policies.py` | two matched keys -> expansions in glossary insertion order, then tuple order; deterministic across runs |
| `expand_query` dedup | `tests/test_domain_policies.py` | shared/duplicate expansion appears once; an expansion already in the query is not re-appended |
| glossary — English attestation | `tests/test_domain_policies.py` | every English expansion string occurs (case-insensitive) in >= 1 `corpus/` file |
| glossary — Spanish renderings | `tests/test_domain_policies.py` | every ES rendering non-empty, distinct from the EN strings; **not** checked against corpus |
| `HybridRetriever` applies expansion | `tests/test_hybrid_retriever_use_case.py` | fake vector/lexical stores receive the expanded string (per config under test); `k` slice unchanged |
| `QueryUseCase` keeps original for prompt | `tests/test_query_use_case.py` | `build_user_prompt` receives the original `question`, not the expanded one |
| regression set frozen | `tests/test_regression_set_integrity.py` | stored SHA-256 matches (mirrors `test_eval_set_integrity`) |
| C2 (if run): embedding vs stored text | vector adapter test | embedder called with heading-prefixed text; `collection.add(documents=...)` gets raw `chunk_text`; `query()` round-trip returns raw `chunk_text` in metadata |

Conventions honoured: port fakes from `tests/fakes.py`, no module-level patching, no real LLM calls.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Expansion lifts a genuinely *unanswerable* query over the gate | Frozen regression set has look-alike ES/EN unanswerables; Phase 1 measures correct-refusal on v1.1.0 per config; any drop below the v1.0.0 baseline (0.900) fails that config. |
| C1 fixes the NPSH incident but not `q009` / decoys | Explicitly expected — hence B and C follow, and acceptance is the whole frozen table, not just the 2 reported queries. |
| Glossary overfits to NPSHA/NPSHR | Corpus-frequency-seeded; EN strings corpus-attested by test; >= 8 entries frozen before measurement; the set cannot grow to chase a result. |
| "ES runs 0.05–0.09 below EN" under-evidenced | Demoted to an open question; the matched-pair gap on v1.1.0's EN/ES counterparts answers it before any per-language threshold. |
| Expanding both channels changes ranking, not just the gate | C1 measured sem-only / bm25-only / both; per-channel English Recall@5 must not regress. |
| Building v1.1.0 after seeing the intervention biases it | v1.1.0 + regression set + acceptance table frozen in Phase 0; a mid-flight eval fix forces a full re-measure. |
| Per-language threshold weakens the "honest refusal" story | Only if the frozen data requires it; SPEC.md disclosure with the same candour as the existing 0.200 note; owner sign-off. |
| C2 reindex changes English retrieval | C2 measured independently; English Recall@5 >= 0.913 hard stop. |
| Touching v1.0.0 reports loses the baseline | Immutability rule (§6 step 5); new output is v1.1.0 with provenance headers. |
| ADR-008 vs `docs/adr/001` layout rules | ADR-008 written and reviewed before C2 lands. |

## 10. Baseline note (resolved as Phase 0 step 1 — required precondition)

`pytest` on `master` (`d2a74aa`) in this dev environment: **184 passed, 2 failed**. Both failures — `test_http_contract_snapshot.py::test_new_app_matches_phase0_snapshot` and `test_http_endpoints.py::test_query_unhandled_exception_returns_500_with_generic_body` — come from a local `.env` setting `API_KEY`, so `/query` returns 401 before the code under test runs. `API_KEY=` unset -> both pass; CI (no `.env`) is green. Real test-isolation gap (these two don't clear/override `API_KEY` like they override the rate limiter). Phase 0 step 1 fixes it as a standalone reviewed change so the baseline is reproducible-green before any intervention.

## 11. Out of scope / follow-ups

- NIOSH decoy-chunk ranking problem (§2 secondary finding) — may be incidentally helped by C2; otherwise its own ticket.
- bge-m3 native multilingual sparse vectors replacing the English-only `\w+` BM25 — larger architectural change, Phase 2b candidate, separate spec.
- Two-tier low-confidence response (§6 Phase 3 option c).
