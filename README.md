# Manufacturing Knowledge RAG Assistant

Portfolio Project 4: a bilingual (EN/ES), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality procedures, with threshold-based refusal when retrieval doesn't support a confident answer.

See [`SPEC.md`](SPEC.md) for full scope, acceptance criteria, no-goals, and the data-honesty/language policy. See [`CLAUDE.md`](CLAUDE.md) for project conventions if you're working on this repo with Claude Code.

**Current status**: Phase 1 (corpus + ingestion) complete. Phases 2 (retrieval) and 3 (generation, UI, deployment) are not started — see `SPEC.md`'s phase status sections.

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

`ingestion/output/chunks.jsonl` (one JSON object per chunk, gitignored/regenerated on every run) is the artifact Phase 2 will read to build embeddings — it is not re-derived by re-parsing the corpus Markdown files.

## Running Tests

```bash
pytest
```

Covers: chunking correctness (target token band, overlap, section-boundary integrity), metadata completeness (every chunk has all required fields — see `ingestion/metadata.py`), and corpus manifest consistency (every file in `corpus/SOURCES.md` exists, every corpus file is listed there with the correct public/synthetic label).
