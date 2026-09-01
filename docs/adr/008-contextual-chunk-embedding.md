# ADR-008: Contextual chunk embedding (`contextual-v1`)

## Status

**Accepted 2026-08-30.** Proposed and written before the C2 implementation;
then measured on the frozen `eval_set` v1.1.0 and approved by the project owner
as the new production index profile
(`docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-results.md` §7–§8).
Adds a second selectable index profile; `raw-v1` stays as the tested rollback
path (`src/features/retrieval/cli.py` still builds it under `INDEX_PROFILE=raw-v1`).

`contextual-v1` / `off` cleared the frozen Spanish Recall@5 ship gate
(0.781 → 0.844) with English Recall@5 flat at 0.917, no regression to the
gate-only refusal proxy or the frozen API-610 regression controls (3/3 still
refused), and a byte-identical citation/prompt payload. The
`expansion_mode=semantic` variant was rejected (API-610 controls 0/3).
`generation_eval` was **not** run — the correct-/false-refusal cells stay
gate-only and unverified. The 8 remaining gate-over-refusals (including the
reported `r001`/`r002`) move to a separate Phase 3 gate-calibration plan.

## Context

The deployed assistant refuses questions the corpus can answer. Phase 1 of the
bilingual / terse-query false-refusal work measured the current index and query
path and produced a committed, machine-reproducible failure classification
(`eval/reports/classification_v1.1.0__raw-v1__off.md`, regenerable with
`python -m src.features.evaluation.failure_classification`; `EXPECTED_COUNTS` in
`src/features/evaluation/failure_classification.py`).

On the answerable subset of `eval/eval_set.json` v1.1.0 (n=80), `raw-v1` /
`expansion_mode=off`:

- **9 same-document-decoy** failures — the expected chunk is absent from top-5,
  but its document is still represented and top-1 comes from that document. The
  retriever finds the right document and picks the wrong section.
- **2 cross-document-decoy** failures (q050, q051) — expected chunk absent,
  expected document still in top-5, top-1 from a different document.
- **0 retrieval-miss** — the expected document is never absent from top-5.
- **15 gate-over-refusal** — the expected chunk *is* retrieved in top-5, but the
  `0.5999` cosine gate refuses. Retrieval is fine; the gate is too strict for
  these (mostly Spanish, mostly terse) queries.

So there are **11 non-gate retrieval failures (9 + 2) and 0 true retrieval
misses**, plus a separate, larger group of **15 gate-over-refusals**.

Spanish Recall@5 on `raw-v1` is **0.781**, below the frozen `>= 0.80` target.
English Recall@5 is **0.913** and is a hard floor — it must not regress.

Root cause of the 11 decoy failures: the vector for each chunk is computed from
its **raw body only**. Sections whose bodies are lexically and semantically
similar (front-matter definitions vs. the substantive data chunk; adjacent
procedure steps; parallel clauses across two OSHA documents) are not separable
by body text alone. The heading path — `document_title`, `section_heading` — is
already stored in chunk metadata but never reaches the embedder.

The embedding model is `BAAI/bge-m3`, pinned to revision
`5617a9f61b028005a4858fdac845db406aefb181` (`MODEL_NAME` / `MODEL_REVISION` in
`src/adapters/secondary/embedder/sentence_transformers_embedder.py`),
`max_seq_length` 8192. The vector store is ChromaDB, collection
`manufacturing_chunks`, cosine space (`hnsw:space=cosine`),
`src/adapters/secondary/vector/chroma_vector_store.py`.

## Decision

Introduce a second index profile, **`contextual-v1`**, that embeds each chunk
with its heading path prepended to the body. `raw-v1` stays as-is and remains
the rollback target and the production default until the owner decides
otherwise.

### Embedding input

The vector input for each chunk becomes:

```
{document_title} > {section_heading}

{chunk_text}
```

Exact delimiter: `f"{document_title} > {section_heading}\n\n{chunk_text}"` — a
space, an ASCII `>`, a space between title and heading; then a blank line
(`\n\n`) before the raw body. This matches the concrete formatter in the design
spec (§6). This prefix is under 30 tokens; `assert_fits_max_seq_length()` runs on
the contextual inputs (not the raw bodies) for `contextual-v1`, against the
bge-m3 `max_seq_length` of 8192.

### What stays raw

- `documents=` passed to Chroma is byte-identical to the source `chunk_text`.
- `metadata["chunk_text"]` is byte-identical to the source `chunk_text`.
- Only the embedding vector input carries the prefix. Retrieval results, the
  generation prompt, and `CitationResolver` all read the raw `chunk_text`. No
  contextual prefix is ever visible to the LLM or in a citation.
- The BM25 lexical index is unchanged — same tokenization, same raw bodies, same
  JSON persistence. `contextual-v1` touches the vector channel only.

This targets the 11 decoy failures. It is **not** expected to fix the 15
gate-over-refusals — those are a threshold / gate-calibration concern deferred
to a possible Phase 3, not something a better ranking can resolve when the chunk
is already retrieved.

### Rebuild, not migrate

`contextual-v1` requires a full reindex via
`python -m src.features.retrieval.cli`. Vectors cannot be recomputed in place;
every chunk's embedding changes. The build reads the same frozen
`ingestion/output/chunks.jsonl` and the same pinned model revision, and records
`MODEL_NAME` / `MODEL_REVISION` plus deterministic `chunks_sha256` /
`corpus_sha256` in `retrieval/output/index_manifest.json`.

### Safe collection replacement (Task 6 implementation plan)

The retrieval CLI is an **offline build**. It must not run concurrently with a
serving process reading the same local Chroma directory.

`build_collection` for `contextual-v1`:

1. Compute and validate the contextual embedding inputs *before* touching the
   live collection.
2. Remove only a known-stale `manufacturing_chunks__candidate` (catching
   Chroma's `NotFoundError`; do not swallow other errors), then build a fresh
   `manufacturing_chunks__candidate` with metadata including
   `hnsw:space=cosine` and `index_profile=contextual-v1`.
3. Add ids, contextual embeddings, raw `documents`, raw metadata to the
   candidate.
4. Require `candidate.count() == len(chunks)` before promotion.
5. Promote by `Collection.modify(name=...)` renames:
   a. remove only a known-stale `manufacturing_chunks__previous`;
   b. rename live `manufacturing_chunks` → `manufacturing_chunks__previous` if it
      exists;
   c. rename candidate → `manufacturing_chunks`;
   d. if step c fails, rename previous back to live and re-raise;
   e. delete `manufacturing_chunks__previous` only after the new live collection
      is confirmed.
6. Build BM25 from the same chunks, then atomically write
   `retrieval/output/index_manifest.json`. If BM25 or manifest writing fails,
   report failure — do not claim the build complete.

### Layout and persistence cross-check

- The change is confined to `src/adapters/secondary/vector/` and
  `src/features/retrieval/`. `src/domain/` stays framework-free (ADR-001, and
  `tests/test_import_invariants.py`); no domain policy, port, or model gains a
  vector-store or embedder dependency. A shared `IndexProfile` literal alias may
  be added to `src/domain/models.py` as a plain type, nothing more.
- No new dependency. The embedding model, vector store, and fusion algorithm are
  unchanged.
- ADR-004's no-`pickle` persistence principle is unaffected: BM25 is not
  touched, and the new manifest is plain JSON.

## Alternatives considered

### Query-side acronym expansion alone (Phase 1, C1) — measured, kept as opt-in

Deterministic corpus-derived acronym expansion applied to the retrieval query
(`expand_query()` + `GLOSSARY` in `src/domain/policies.py`;
`HybridRetriever(expansion_mode=...)`). Measured in Phase 1: `semantic` mode
fixed the reported NPSHA/NPSHR queries but left Spanish Recall@5 at **0.781**
(< 0.80) and flipped two API-610 decoy control queries from refuse → answer.
Retained as an unflippable-in-production opt-in — `expansion_mode` is not a
`Settings` field and is not env-overridable, so production is hard-wired to
`"off"`. C1 alone does not clear the frozen acceptance table, which is why C2 is
proposed.

### Adding more corpus documents — out of scope

No corpus changes this phase. Phase 1 classification shows **0** true retrieval
misses — the expected document is always retrieved. This is a ranking problem
within already-indexed content, not a coverage gap. Corpus addition stays
deferred (a separate phase, gated on evidence of a real content gap).

### Lowering `REFUSAL_COSINE_THRESHOLD` to rescue ranking misses — rejected

A threshold change cannot fix a ranking problem. Lowering `0.5999` only trades
false-refusals for false-answers and does nothing for a chunk that is not in
top-5. `0.5999` is a documented byte-stable invariant. Gate calibration
(including an evidence-supported per-language threshold or a two-tier
low-confidence response) is a separate Phase 3 concern with its own owner
sign-off — it is the right home for the 15 gate-over-refusals, not for the 11
decoys.

### Replacing or augmenting BM25 with multilingual sparse retrieval — deferred

bge-m3's native multilingual sparse vectors replacing the English-only `\w+`
BM25 tokenizer is a larger architectural change. Deferred to a possible Phase 2b
with its own plan.

## Consequences

- A new index profile identifier, `contextual-v1`, joins `raw-v1`. Evaluation
  artifacts are named `*_v1.1.0__<index_profile>__<expansion_mode>.*`; the
  canonical unsuffixed baseline stays `raw-v1` / `off`.
- Every future reindex must choose a profile. The manifest records which one
  built the live collection; `_eval_retriever` verifies the expected manifest
  before measuring.
- Retrieval, citation, and prompt behavior are byte-identical to `raw-v1` for
  any query — only similarity scores and therefore ranking move. This is
  verifiable: stored `documents` and `metadata["chunk_text"]` round-trip raw.
- English Recall@5 `>= 0.913` is a hard stop for adopting `contextual-v1`. If
  the contextual index regresses English retrieval, refusal precision, or
  citation payload, it is rejected and `raw-v1` stays live.
- The 15 gate-over-refusals are expected to persist. The Phase 2 decision
  package must report them and route them to a Phase 3 gate plan, not present
  `contextual-v1` as a full fix.
- `contextual-v1` may incidentally help the NIOSH front-matter decoy-ranking
  problem noted as a Phase 1 secondary finding; that is a hoped-for side effect,
  not a commitment.

## Rollback

Rebuild `raw-v1` from the same frozen `ingestion/output/chunks.jsonl` and the
same pinned model revision (`5617a9f61b028005a4858fdac845db406aefb181`) via
`INDEX_PROFILE=raw-v1 python -m src.features.retrieval.cli` (the bare command now
builds `contextual-v1`). The `raw-v1` build path is kept and tested for exactly
this reason — it still embeds raw bodies.

If promotion fails mid-swap, `manufacturing_chunks__previous` is renamed back to
live and the error re-raised; the previous live collection stays queryable and
unchanged.

Never mutate `eval/eval_set.json`, never regenerate a v1.0.0 report, and never
lower the acceptance targets in the frozen table to make `contextual-v1` pass.
