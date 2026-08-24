# Manufacturing Knowledge RAG Assistant

Portfolio Project 4: a bilingual (EN/ES), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality procedures, with threshold-based refusal when retrieval doesn't support a confident answer.

See [`SPEC.md`](SPEC.md) for full scope, acceptance criteria, no-goals, and the data-honesty/language policy. See [`CLAUDE.md`](CLAUDE.md) for project conventions if you're working on this repo with Claude Code.

**Current status**: Phase 1 (corpus + ingestion) and Phase 2 (retrieval) complete. Phase 3 (generation, UI, deployment) is not started — see `SPEC.md`'s phase status sections.

## Corpus

14 documents under `corpus/`: 9 real, public-domain U.S. government documents (OSHA publications, DOE Fundamentals Handbooks, NIOSH guidance, CFR regulatory text) and 5 clearly-labeled synthetic documents filling gaps public sources don't cover. Every document is listed in [`corpus/SOURCES.md`](corpus/SOURCES.md) with its exact source and public/synthetic label.

## Running Ingestion Locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m ingestion.run
```

The first run needs network access once, to let `tiktoken` download and cache its `cl100k_base` encoding file; subsequent runs work offline. If that download fails with no network available, `python -m ingestion.run` will raise a clear error saying so rather than an opaque one.

Expected output:

```
Documents processed: 14 (9 public, 5 synthetic)
Chunks produced: <N>
Total corpus size: <words> words, <chars> characters
Chunks written to: .../ingestion/output/chunks.jsonl
```

`ingestion/output/chunks.jsonl` (one JSON object per chunk, gitignored/regenerated on every run) is the artifact Phase 2 reads to build embeddings — it is not re-derived by re-parsing the corpus Markdown files.

## Running Retrieval Evaluation Locally

Requires `ingestion/output/chunks.jsonl` to already exist (run ingestion first, above).

```bash
python -m retrieval.build_index
```

Embeds all chunks with `BAAI/bge-m3` (multilingual, CPU-only) and builds both the ChromaDB vector store and the BM25 lexical index under `retrieval/output/` (gitignored/regenerated on every run). The first run needs network access to download and cache the ~2.3GB model; subsequent runs work offline (`HF_HUB_OFFLINE=1` speeds this up once cached). Expected output:

```
Chunks embedded: 228
Embedding model: BAAI/bge-m3 (max_seq_length=8192)
Vector store: .../retrieval/output/chroma (collection 'manufacturing_chunks')
BM25 index: .../retrieval/output/bm25_index.pkl
```

Then run the evaluation:

```bash
python -m eval.run_eval
```

Verifies `eval/eval_set.json`'s SHA-256 hash, runs all 40 hand-written questions (30 answerable — including 7 in Spanish against the English corpus, 10 deliberately unanswerable) through the hybrid retriever, and writes a versioned report to `eval/reports/retrieval_report_v<version>.md` with recall@3, recall@5, MRR, a per-language breakdown, and example queries. See `SPEC.md`'s Phase 2 Status for the latest results (recall@5 = 0.833, MRR = 0.621).

To edit the eval set, change `eval/eval_set.json`, bump its `version` field, then regenerate the hash:

```bash
python -m eval.hash_eval_set --write
```

## Running Tests

```bash
pytest
```

Covers: chunking correctness (target token band, overlap, section-boundary integrity), metadata completeness (every chunk has all required fields — see `ingestion/metadata.py`), corpus manifest consistency (every file in `corpus/SOURCES.md` exists, every corpus file is listed there with the correct public/synthetic label), the embedding model's fit against real corpus chunk sizes, hybrid retriever fusion correctness, and eval-set integrity (hash, split, expected chunk IDs all exist).
