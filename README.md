# Manufacturing Knowledge RAG Assistant

Portfolio Project 4: a bilingual (EN/ES), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality procedures, with threshold-based refusal when retrieval doesn't support a confident answer.

See [`SPEC.md`](SPEC.md) for full scope, acceptance criteria, no-goals, and the data-honesty/language policy. See [`CLAUDE.md`](CLAUDE.md) for project conventions if you're working on this repo with Claude Code.

**Current status**: Phase 1 (corpus + ingestion) and Phase 2 (retrieval) complete. Phase 3 (generation, UI, deployment) is implemented — the FastAPI backend, Streamlit UI, and Docker container all exist and are tested — but **not yet deployed**: no live Hugging Face Spaces instance exists yet. See `SPEC.md`'s phase status sections, including the Phase 3 honest-disclosure note on the refusal threshold gate.

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

`requirements.txt` pins floors (`>=`), not exact versions — deliberate, see the comment in that file. For a reproducible install matching a known-good, fully-tested dependency tree, use `requirements-lock.txt` instead (`pip install -r requirements-lock.txt`); regenerate it with `pip freeze > requirements-lock.txt` after any `requirements.txt` change.

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

Verifies `eval/eval_set.json`'s SHA-256 hash, runs all 40 hand-written questions (30 answerable — including 7 in Spanish against the English corpus, 10 deliberately unanswerable) through the hybrid retriever, and writes a versioned report to `eval/reports/retrieval_report_v<version>.md` with recall@3, recall@5, MRR, a per-language breakdown, and example queries. See `SPEC.md`'s Phase 2 Status for the latest results (recall@5 = 0.833, MRR = 0.637).

To edit the eval set, change `eval/eval_set.json`, bump its `version` field, then regenerate the hash:

```bash
python -m eval.hash_eval_set --write
```

To sweep and pick the refusal cosine-similarity threshold used by Phase 3:

```bash
python -m eval.threshold_analysis
```

Writes `eval/reports/threshold_analysis_v<version>.md`. See `SPEC.md`'s Phase 3 Status for an honest note on what this threshold does and doesn't guarantee on its own.

## Running the Generation API and UI Locally

Requires `retrieval/output/` to already exist (run `python -m retrieval.build_index` first, above) and a `.env` file (copy `.env.example`) with at least a `GROQ_API_KEY` set (`OPENAI_API_KEY` is used as fallback only). `API_KEY` is optional: if set, `/query` requires the same value in the request's `X-API-Key` header (the Streamlit UI reads and sends it automatically from its own environment); leave it unset for local development.

Run the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Serves `GET /health` and `POST /query` on `http://localhost:8000`.

In a second terminal, with the backend already running, run the Streamlit UI:

```bash
streamlit run ui/streamlit_app.py
```

Reads the `API_BASE_URL` env var (default `http://localhost:8000`) to reach the backend.

To evaluate end-to-end generation quality (retrieval + refusal + LLM generation + citation resolution) against the eval set:

```bash
python -m eval.run_generation_eval
```

Makes real LLM API calls for all 40 eval questions with a deliberate 20-second delay between each (to stay under Groq's free-tier rate limits) — **expect this to take roughly 13-15 minutes**, and expect real (small) LLM API spend. Writes `eval/reports/generation_eval_v<version>.md` (correct-refusal rate, false-refusal rate, latency summary) and `eval/reports/manual_review_checklist_v<version>.csv` (citation accuracy and faithfulness are graded by hand, not by this script — see the report's own "Manual Review Required" section).

## Running the Docker Build Locally

The combined-container deploy image (FastAPI + Streamlit + nginx in one container, matching the Hugging Face Spaces Docker SDK target):

```bash
docker build -t rag4 .
docker run -p 7860:7860 rag4
```

The build runs ingestion and builds the retrieval index (embedding all chunks with `bge-m3`) as part of the image build itself, so **the first build is slow — expect it to take 10+ minutes**, most of it downloading and embedding with `bge-m3`'s ~2.3GB weights; subsequent builds only redo this if the corpus or index-build code changes.

**Not verified end-to-end in this repo's dev environment** — Docker Desktop's WSL2 backend was unavailable in the sandbox this branch was built in, so the build/run has not actually been exercised here. The Dockerfile's non-root-user handling (required for Hugging Face Spaces, which runs containers as uid 1000) was fixed by static reasoning about Debian/Docker/HF-Spaces semantics, not confirmed via a real build — treat it as the first thing to test once a real Docker daemon is available (locally, or during actual Spaces deployment).

**Hugging Face Space config note:** a Space's `sdk`/`app_port`/etc. configuration comes from a YAML frontmatter block at the very top of the `README.md` sitting at the root of the git repository backing that Space. If this project is deployed the standard way — adding the Space as a second git remote and pushing this same repo's content to it — then that root `README.md` is *this file*, meaning the required `sdk: docker` / `app_port: 7860` frontmatter block will need to be added to the top of this exact file immediately before that push (it is deliberately not added now, since it isn't accurate until a Space actually exists to push to, and premature frontmatter here would misrepresent this repo's current, undeployed state). If a separate mirror repo is used for the Space instead, the same frontmatter requirement applies to that repo's root `README.md` instead. Either way, this is a deploy-time step, not something to pre-populate speculatively before the Space exists.

## Running Tests

```bash
pytest
```

Covers: chunking correctness (target token band, overlap, section-boundary integrity), metadata completeness (every chunk has all required fields — see `ingestion/metadata.py`), corpus manifest consistency (every file in `corpus/SOURCES.md` exists, every corpus file is listed there with the correct public/synthetic label), the embedding model's fit against real corpus chunk sizes, hybrid retriever fusion correctness, eval-set integrity (hash, split, expected chunk IDs all exist), refusal threshold logic, LLM client JSON repair/retry/fallback behavior (provider SDKs mocked — no real LLM calls), generation orchestration and citation resolution, the FastAPI endpoints, and the Streamlit UI's pure logic functions.

## Linting and Type-Checking

```bash
pip install ruff mypy
ruff check .
mypy api ingestion retrieval eval ui
```

Config lives in `pyproject.toml`. `mypy` is scoped to the application code, not `tests/`.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `master`: installs `requirements-lock.txt`, runs `ruff check .` and `mypy`, verifies the eval set's SHA-256 hash (`python -m eval.hash_eval_set --verify`), then runs the full `pytest` suite — including the real `bge-m3` model load in `tests/test_embedding_model_fits_corpus.py`, cached across runs via `actions/cache` to avoid re-downloading the ~2.3GB weights every time.
