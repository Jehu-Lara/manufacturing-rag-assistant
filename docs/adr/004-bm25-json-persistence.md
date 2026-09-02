# ADR-004: BM25 index persisted as JSON, not pickle

## Status

**Accepted.** In force, and load-bearing: the no-pickle rule is one of the repo's
prohibited patterns, not merely a past preference.

## Context

`BM25Okapi` (from `rank_bm25`) has no native serialization. The original
`retrieval/bm25_index.py` pickled the fitted model object directly —
functionally correct (the file was always generated and read only by that
same module, from this project's own gitignored, locally-regenerated build
output, never from an untrusted source), but `pickle.load`/`pickle.dump` in
runtime code is exactly the kind of pattern this refactor's DoD forbids, and
it blocks treating the lexical index like the rest of the ports/adapters
architecture (a `Protocol`-shaped adapter, constructed with a plain data
path, no special-cased deserialization risk).

## Decision

`src/adapters/secondary/lexical/bm25_lexical_index.py` persists
`{chunk_ids, corpus_tokens}` as plain JSON — the tokenized corpus, not the
fitted model. On load, `BM25Okapi(corpus_tokens)` is rebuilt in memory (a
single term-frequency pass) and cached on the adapter instance for the
process lifetime, replacing the old module-level `_CACHE` singleton.

An existing `.pkl` is not converted. Verified directly against the
installed `rank_bm25==0.2.2`: `BM25Okapi.doc_freqs` stores per-document
word→count dictionaries (order-irrelevant to BM25's math), so a
technically-correct token-multiset reconstruction from a pickled instance
*is* possible — but the `.pkl` file is gitignored and already trivially
regenerable from `chunks.jsonl` via the ingestion+index-build CLI chain, so
writing a one-time migration path for a disposable file wasn't worth it.
Rebuild-via-CLI is the documented path instead.

## Consequences

- Zero `pickle.load`/`pickle.dump` in this project's own runtime code
  (verified — grep for `pickle` across `src/adapters/secondary/lexical/`
  returns nothing).
- **Scope caveat, stated explicitly here**: this eliminates pickle from
  *our own* code only. `chromadb` itself may still use pickle internally in
  its own persistence layer (outside this project's control) — a
  third-party supply-chain consideration this ADR does not claim to have
  resolved. "No pickle" describes this project's adapters, not the full
  dependency tree.
- Larger on-disk file than pickle's compact fitted object (token lists
  repeat per document) — acceptable at this project's portfolio scale, not
  production scale.
- One-time rebuild cost per process start (a single `BM25Okapi(...)` call),
  not per query — same caching behavior as before, just instance-scoped.
