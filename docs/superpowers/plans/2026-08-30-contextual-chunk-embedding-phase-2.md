# Bilingual / terse-query false-refusal — Phase 2 contextual embedding plan

> **Execution rule:** implement one task at a time, preserve the frozen eval sets, and stop at every explicit owner gate. This plan is the fallback equivalent of `writing-plans`; that skill was not available in this workspace.

**Goal:** test whether heading-aware chunk embeddings resolve the measured decoy-ranking failures and raise Spanish Recall@5 from `0.781` to at least `0.80`, without regressing English retrieval, refusal precision, citation integrity, or the API-610 controls.

**Hypothesis:** embedding `document_title › section_heading\n\nchunk_text` will distinguish sections whose raw bodies are semantically similar. It is not assumed to pass: Phase 1 classified 10 failures as same-document/wrong-section decoys and 2 as cross-document decoys. The frozen evidence decides.

**Architecture:** Chroma computes vectors from contextualized inputs but stores the original `chunk_text` in both `documents=` and metadata. BM25, RRF (`k=60` and ascending `chunk_id` tie-break), prompt construction, citations, and `REFUSAL_COSINE_THRESHOLD=0.5999` remain unchanged. Evaluation artifacts identify two independent axes: `index_profile` and `expansion_mode`.

**Dataset identity:** `eval_set` remains **v1.1.0** and the regression set remains **v1.0.0**. Report names must not imply a new dataset version.

**Scope:** RAG4 only. No corpus addition, embedding-model change, sparse-retrieval replacement, threshold change, production-default expansion change, deploy, PR merge, or push.

---

## Global constraints

- Start only after the Phase 1 closeout commit exists on `fix/bilingual-refusal`: real `generation_eval` results for `off` and `semantic`, canonical unsuffixed baseline = `off`, and the classification corrected to 10 same-document + 2 cross-document decoys.
- Treat `eval/eval_set.json`, `eval/regression_queries.json`, and every v1.0.0/v1.1.0 baseline report as immutable inputs. Do not edit questions, expected chunks, hashes, or acceptance thresholds.
- New report names use `*_v1.1.0__<index_profile>__<expansion_mode>.*`, where `index_profile in {raw-v1, contextual-v1}` and `expansion_mode in {off, semantic}` for this phase.
- Preserve the existing canonical aliases produced in Phase 1; do not overwrite them while measuring Phase 2.
- `documents=` and `metadata["chunk_text"]` remain raw and byte-identical to the input chunk. Only the vector input changes.
- `src/domain/` remains framework-free. No new dependency is allowed.
- Tests make no real LLM calls. Paid `generation_eval` runs require an explicit owner gate immediately before execution.
- Generated retrieval indexes stay under ignored `retrieval/output/`. Only manifests copied into report provenance and evaluation artifacts are committed.
- No `.env`, secret value, prompt body, generated answer, or provider response is written to the new JSONL retrieval-details artifact.
- Do not commit, push, merge, or deploy during Tasks 1-8. Task 9 presents the evidence and waits for owner authorization.

## Frozen acceptance table

| Metric | Required outcome |
|---|---|
| English Recall@5, answerable | `>= 0.913` hard stop |
| Spanish Recall@5, answerable | `>= 0.80` to ship; otherwise ranking follow-up or approved exception |
| English correct-refusal | `>= 0.90` |
| Spanish correct-refusal | `>= 0.80` |
| English false-refusal | `<= 0.10` |
| Spanish false-refusal | material improvement with explicit disclosure |
| Reported NPSHA/NPSHR en + es | correct cited answers |
| Regression unanswerable controls | no regression versus the frozen Phase 1 baseline; report API-610 rows individually |
| Citation/prompt payload | raw `chunk_text`; no contextual prefix visible |

The matched-pair EN-ES cosine gap is evidence, not a pass/fail gate. A language-specific threshold remains a Phase 3 option only if stratified distributions and error patterns support it.

---

## Files expected to change

| File | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md` | Phase 1 closeout correction and later Phase 2 comparison |
| `docs/adr/008-contextual-chunk-embedding.md` | decision, alternatives, rollback, consequences |
| `src/domain/models.py` | `IndexProfile` literal alias if a shared type is needed |
| `src/features/retrieval/index_manifest.py` | deterministic corpus/chunks hashes, build identity, manifest read/write/verify |
| `src/features/retrieval/cli.py` | build profile selection, contextual length check, manifest emission |
| `src/adapters/secondary/vector/chroma_vector_store.py` | contextual embedding inputs, raw stored documents, candidate collection swap |
| `src/features/evaluation/artifacts.py` | shared artifact suffix and provenance header helpers |
| `src/features/evaluation/retrieval_eval.py` | config-aware names plus JSONL top-5 details |
| `src/features/evaluation/threshold_analysis.py` | config-aware retriever and report names |
| `src/features/evaluation/regression_eval.py` | profile/config-aware report output |
| `src/features/evaluation/generation_eval.py` | collision-safe profile/config names; canonical alias only when explicitly requested |
| `src/features/evaluation/_eval_retriever.py` | verify expected index manifest before measuring |
| `tests/test_index_manifest.py` | manifest hashing, build identity, mismatch tests |
| `tests/test_evaluation_artifacts.py` | suffix/provenance contracts |
| `tests/test_evaluation_retrieval_eval.py` | JSONL shape, top-5 order, filenames |
| `tests/test_evaluation_generation_eval.py` | no overwrite across configs |
| `tests/test_chroma_vector_store.py` | contextual input, raw storage, candidate/rollback behavior |
| `tests/test_retrieval_cli.py` | contextual max-length and manifest integration |

Exact file names may be consolidated if the same responsibility already has a natural tested home; do not create parallel helpers with overlapping behavior.

---

## Task 0 — Verify Phase 1 is closed

**Changes:** none.

- [ ] Run `git status --short --branch` and record HEAD. Stop if the working tree contains unexplained files or if the background evaluation is still writing artifacts.
- [ ] Confirm the Phase 1 closeout commit contains:
  - `generation_eval_v1.1.0__off.md` and `generation_eval_v1.1.0__semantic.md`;
  - `generation_eval_v1.1.0.md` byte-equal to the `off` baseline;
  - corresponding manual-review CSV names that do not overwrite each other;
  - real, not gate-only, correct-/false-refusal numbers in the results document;
  - classification wording `10 same-document + 2 cross-document`, not `12 inside the correct document`.
- [ ] Run both integrity guards:
  - `.\.venv\Scripts\python.exe -m src.features.evaluation.eval_set_integrity --verify`
  - `.\.venv\Scripts\python.exe -m src.features.evaluation.regression_set_integrity --verify`
- [ ] Stop and ask the owner if any precondition is absent. Do not repair Phase 1 inside the C2 increment.

Expected: clean, closed Phase 1 baseline with immutable artifacts.

---

## Task 1 — Define artifact and index-manifest contracts (tests first)

**Files:** create `tests/test_evaluation_artifacts.py`, `tests/test_index_manifest.py`; then create the minimal implementation modules.

- [ ] Write failing tests for `artifact_suffix(index_profile, expansion_mode)`:
  - `raw-v1/off -> "__raw-v1__off"`;
  - `contextual-v1/semantic -> "__contextual-v1__semantic"`;
  - reject unknown values with `ValueError` rather than silently producing a filename.
- [ ] Write failing tests for a deterministic provenance header containing:
  - eval version + stored hash;
  - regression-set version + stored hash;
  - `index_profile`, `expansion_mode`, threshold;
  - `chunks_sha256`, `corpus_sha256`, embedding model and revision, build commit;
  - evaluation commit separately from index build commit.
- [ ] Write failing tests for deterministic hashes:
  - `chunks_sha256` hashes the exact bytes of `ingestion/output/chunks.jsonl`;
  - `corpus_sha256` hashes sorted relative `.md` paths plus file bytes, so renaming a file changes the hash;
  - `.env`, PDFs, ignored output, and filesystem timestamps are never included.
- [ ] Write failing tests for build-commit resolution in this order: explicit argument, `DEPLOYED_SHA`, Git HEAD, then literal `"unknown"`. Never fail a Docker build merely because `.git` is absent.
- [ ] Implement the smallest pure helpers needed to pass. Use frozen dataclasses or typed immutable values; no module-level mutable cache.
- [ ] Run:
  - `.\.venv\Scripts\python.exe -m pytest tests\test_evaluation_artifacts.py tests\test_index_manifest.py -q`
  - `ruff check src tests`
  - `mypy src`

Expected: deterministic, secret-free artifact identity with distinct index-build and evaluation commits.

---

## Task 2 — Make report naming collision-safe

**Files:** evaluation runners and their existing tests.

- [ ] Add fail-first tests showing two sequential `generation_eval.run()` calls for `raw-v1/off` and `raw-v1/semantic` produce different Markdown and CSV paths.
- [ ] Add equivalent path tests for retrieval, threshold, and regression outputs. Preserve the dataset version `v1.1.0` in every name.
- [ ] Add explicit `index_profile` and `expansion_mode` parameters to evaluation entry points. Keep default behavior backward-compatible, but Phase 2 calls must pass both explicitly.
- [ ] Make canonical unsuffixed aliases opt-in (`write_canonical_alias=True`) and legal only for `raw-v1/off`. A contextual or semantic run must never overwrite the canonical baseline.
- [ ] Route all generated headers through the Task 1 helper; remove hand-authored provenance from newly generated Phase 2 files.
- [ ] Run the focused evaluation tests, then `ruff` and `mypy`.

Expected: no wrapper is required to prevent overwrites, and filenames state dataset + index + query configuration truthfully.

---

## Task 3 — Emit machine-readable retrieval details and revalidate 10+2

**Files:** `retrieval_eval.py`, tests, generated JSONL artifacts, results document.

- [ ] Add a fail-first test for one JSON object per question with exactly:

```json
{"id":"q001","lang":"en","top5":[{"chunk_id":"...","rank":1,"semantic_score":0.0,"semantic_rank":1,"bm25_rank":2,"fused_score":0.0}],"gate_decision":"answer"}
```

- [ ] Preserve top-5 fused order. Represent absent semantic/BM25 ranks and scores as JSON `null`, never invented zeros.
- [ ] Include expected chunk/document identifiers only if needed for deterministic classification; do not include query text, generated answers, prompts, or secrets.
- [ ] Write JSONL atomically (`.tmp` in the same directory, then `Path.replace`) so interruption cannot leave a valid-looking partial artifact.
- [ ] Add a pure classifier with mutually exclusive outputs:
  - `gate-over-refusal`: expected chunk in top 5, gate refuses;
  - `same-document-decoy`: expected chunk absent, expected document represented, top result belongs to that document;
  - `cross-document-decoy`: expected chunk absent, expected document represented, top result belongs to another document;
  - `retrieval-miss`: expected document absent from top 5.
- [ ] Generate `retrieval_details_v1.1.0__raw-v1__off.jsonl` from the existing raw index and regenerate the classification table from this artifact, not manual notes.
- [ ] Assert the frozen baseline yields 10 same-document + 2 cross-document non-gate failures and zero expected-document-absent misses. If counts differ, stop; do not edit the eval set to restore them.

Expected: the Phase 1 diagnosis is reproducible from committed machine-readable evidence.

---

## Task 4 — Record ADR-008 before C2 code

**File:** `docs/adr/008-contextual-chunk-embedding.md`.

- [ ] Document context: raw-body embeddings, 10 same-document decoys, 2 cross-document decoys, ES Recall@5 `0.781`.
- [ ] Record the decision:
  - vector input = `document_title › section_heading\n\nchunk_text`;
  - `documents=` and metadata stay raw;
  - BM25 stays raw and unchanged;
  - profile id = `contextual-v1`;
  - full rebuild required.
- [ ] Record alternatives rejected/deferred: query expansion alone, corpus addition, threshold change for ranking misses, sparse multilingual retrieval.
- [ ] Record rollback: rebuild `raw-v1` from the same frozen chunks/model revision; do not mutate the eval set or lower acceptance targets.
- [ ] Record operational safety: candidate collection, count validation, rename swap, rollback to previous collection on swap failure. The CLI is an offline build; do not run it concurrently with the serving process.
- [ ] Cross-check ADR-001 layout/import constraints and ADR-004 persistence principles.

Expected: decision and rollback are reviewable before implementation.

---

## Task 5 — Add fail-first C2 and rebuild-safety tests

**Files:** `tests/test_chroma_vector_store.py`, `tests/test_retrieval_cli.py` and minimal fixtures.

- [ ] Build a two-chunk fixture with distinct titles/headings and raw bodies.
- [ ] With a recording fake `EmbedderPort`, assert `contextual-v1` receives exactly the prefixed strings, in the same order as ids.
- [ ] Inspect Chroma directly with `collection.get(ids=[...], include=["documents", "metadatas"])` and assert:
  - stored document equals raw `chunk_text` byte-for-byte;
  - `metadata["chunk_text"]` equals raw `chunk_text`;
  - neither stored value contains the prefix unless it already existed in the source body.
- [ ] Assert `raw-v1` still embeds raw bodies so rollback remains executable and tested.
- [ ] Assert `assert_fits_max_seq_length()` receives contextual inputs for `contextual-v1`, not raw bodies.
- [ ] Seed a valid live collection, make the candidate build fail, and assert the live collection remains queryable and unchanged.
- [ ] Assert a stale candidate is removed before a new candidate is created, a successful candidate contains exactly `len(chunks)` rows before the swap, and the previous collection is cleaned only after success.
- [ ] Run the focused tests and confirm they fail for the intended missing behavior, not fixture errors.

Expected: failures characterize raw-storage integrity and recovery risk before production code changes.

---

## Task 6 — Implement contextual-v1 and safe collection replacement

**Files:** vector adapter, retrieval CLI, manifest helper; no query/prompt/citation files.

- [ ] Add one pure formatter for contextual embedding inputs. Do not duplicate the delimiter in CLI/tests.
- [ ] In `build_collection`, compute and validate embedding inputs before touching the live collection.
- [ ] Remove only a known stale `__candidate` collection (catching `NotFoundError`), then build a fresh candidate with metadata including `hnsw:space=cosine` and `index_profile=contextual-v1`.
- [ ] Add ids, contextual embeddings, raw `documents`, and raw metadata to the candidate.
- [ ] Require `candidate.count() == len(chunks)` before promotion.
- [ ] Promote safely using the installed Chroma API `Collection.modify(name=...)`:
  1. remove only a known stale `__previous` collection;
  2. rename live to `__previous` if it exists;
  3. rename candidate to live;
  4. if step 3 fails, rename previous back to live and re-raise;
  5. delete previous only after the live collection is confirmed.
- [ ] Catch only Chroma's `NotFoundError` for expected absence. Do not swallow arbitrary build, add, or rename errors.
- [ ] Build BM25 from the same chunks, then atomically write `retrieval/output/index_manifest.json`. If BM25 or manifest writing fails, report failure and do not claim the build complete.
- [ ] Include `MODEL_NAME` and `MODEL_REVISION`; never infer revision from a mutable remote tag.
- [ ] Run focused tests, `ruff`, and `mypy`.

Expected: contextual vectors with raw prompt/citation payload and recoverable rebuild failure.

---

## Task 7 — Reindex contextual-v1 and verify the physical artifact

**Changes:** ignored retrieval output only.

- [ ] Confirm no app process is serving from the local Chroma directory.
- [ ] Capture the existing raw manifest/report references; do not delete committed baseline reports.
- [ ] Run `.\.venv\Scripts\python.exe -m src.features.retrieval.cli` and retain the complete console output.
- [ ] Verify:
  - manifest profile = `contextual-v1`;
  - manifest chunks/corpus hashes match current frozen inputs;
  - collection count equals chunk count;
  - collection metadata profile matches the manifest;
  - stored sample documents and metadata remain raw;
  - BM25 artifact exists and loads.
- [ ] Run the real-corpus max-sequence-length test.
- [ ] If reindex fails, verify the previous live collection remains usable and stop. Do not lower limits, truncate, or delete the old index.

Expected: one internally consistent contextual-v1 index; no metric claim yet.

---

## Task 8 — Measure contextual-v1 off and semantic

**Outputs:** config-suffixed Markdown/CSV/JSONL artifacts; no canonical overwrite.

- [ ] For `contextual-v1/off`, run:
  - retrieval evaluation;
  - threshold analysis (diagnostic only; no threshold application);
  - regression evaluation;
  - JSONL detail/classification generation.
- [ ] Repeat for `contextual-v1/semantic`.
- [ ] Before any LLM call, present retrieval/regression results to the owner. Stop immediately if English Recall@5 `< 0.913`, hashes mismatch, API-610 controls regress beyond baseline, or the reported NPSH target chunk disappears.
- [ ] With explicit approval, run `generation_eval` for both configs. Record provider/model, index manifest, expansion mode, evaluation commit, errors, and run completion; never record key material.
- [ ] Do not compare a partial/error-heavy generation run as if complete. Define completeness as 105 terminal rows with no unclassified execution errors; otherwise mark the config unverified.
- [ ] Run both integrity guards after measurement and verify every v1.0.0/v1.1.0 baseline file hash remains unchanged.

Expected: complete comparable evidence for raw baseline, contextual-only, and contextual+semantic.

---

## Task 9 — Decision package and owner gate

**Files:** results document, ADR status, SPEC/CLAUDE only after the owner chooses SHIP.

- [ ] Produce one comparison table with raw-v1/off, raw-v1/semantic, contextual-v1/off, contextual-v1/semantic.
- [ ] Report per language: Recall@3/@5, MRR, correct-refusal, false-refusal, matched-pair gap, NPSH outcomes, API-610 controls, and failure classification counts.
- [ ] Apply decision branches exactly:
  - **SHIP candidate:** all hard cells pass. Recommend either contextual-only or contextual+semantic from measured evidence.
  - **Ranking follow-up:** ES Recall@5 remains `<0.80` or expected-document-absent misses remain. Consider multilingual sparse retrieval in a separate Phase 2b plan, or request an explicit documented exception; do not route this to threshold calibration.
  - **Gate follow-up:** retrieval passes but gate-over-refusals remain. Open a separate Phase 3 plan for global-threshold analysis, two-tier response, or evidence-supported language-specific thresholds.
  - **Reject/rollback:** English regression, refusal-precision regression, citation/payload change, manifest mismatch, or unresolved API-610 regression.
- [ ] Present the fixed review format:

```text
VEREDICTO: PASS | PASS_WITH_FIXES | FAIL
SEVERIDAD: P0 | P1 | P2
BLOCKERS:
FIXES_REQUIRED:
RESIDUAL_RISKS:
```

- [ ] Stop for explicit owner confirmation. Only after SHIP approval may a follow-up increment update production defaults, SPEC.md, CLAUDE.md, and ADR status.
- [ ] Deployment remains a separate `DEPLOY`-gated operation even after merge approval.

Expected: evidence-backed owner decision, with no silent threshold/default/deployment change.

---

## Final verification gate

Run separately and capture each result:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
ruff check src tests
mypy src
.\.venv\Scripts\python.exe -m src.features.evaluation.eval_set_integrity --verify
.\.venv\Scripts\python.exe -m src.features.evaluation.regression_set_integrity --verify
git diff --check
git status --short --branch
```

Then review `git diff` for secrets, `.env`, PDFs, query text in JSONL, accidental baseline rewrites, and documentation claims not supported by generated artifacts.

## Explicit non-actions

- Do not edit or regenerate v1.0.0 reports.
- Do not bump eval-set v1.1.0.
- Do not change `REFUSAL_COSINE_THRESHOLD`, RRF, the embedding model/revision, BM25 tokenization, corpus contents, or citation resolution.
- Do not enable `expansion_mode=semantic` in production merely because NPSH passes.
- Do not commit, push, merge, or deploy without the owner gate defined above.
