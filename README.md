# Manufacturing Knowledge RAG Assistant

Portfolio Project 4: a bilingual (EN/ES), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality procedures, with threshold-based refusal when retrieval does not support a confident answer.

See [`SPEC.md`](SPEC.md) for the complete scope, decisions, measured limitations, and data-honesty policy. See [`CLAUDE.md`](CLAUDE.md) for current contributor commands and architecture rules.

## Current status

- Corpus and ingestion: complete — 14 documents (9 public, 5 clearly labeled synthetic) produce 228 metadata-complete chunks.
- Retrieval: complete — hybrid BM25 + `BAAI/bge-m3` retrieval; Recall@5 is 0.833.
- Generation/API/UI: implemented and tested — FastAPI, Streamlit, Groq (`openai/gpt-oss-120b`) with OpenAI (`gpt-4o-mini`) fallback, bilingual responses, deterministic refusal gate, and citations resolved from retrieved metadata.
- Evaluation: correct-refusal 0.900 (passes), false-refusal 0.200 (documented exception), citation accuracy 23/30 = 0.767 (below target), and faithfulness 29/30 = 0.967 (passes). The citation and faithfulness verdicts are human-reviewed, not LLM-as-judge scores.
- Public showcase: live as a free [Hugging Face Static Space](https://huggingface.co/spaces/JehuLara/manufacturing-rag-assistant). It is a portfolio page only and does not run the RAG.
- Interactive deployment: a second, separate Hugging Face PRO Docker Space (amd64, CPU Basic), published manually from a selected green `master` commit through a keyless GitHub Actions workflow. Oracle Cloud was the intended runtime until 2026-08-28, when it was abandoned after remaining blocked on regional ARM capacity — see [`docs/adr/007-hf-pro-docker-space-deploy.md`](docs/adr/007-hf-pro-docker-space-deploy.md). No live interactive deployment exists yet.

## Architecture

All application code lives under `src/` as a modular monolith with ports and adapters:

- `src/domain/`: framework-free models, ports, RRF/refusal policy, and citation resolution.
- `src/features/`: ingestion, retrieval, query, and evaluation use cases.
- `src/adapters/`: FastAPI and concrete ChromaDB, BM25, embedding, and LLM adapters.
- `src/web/`: Streamlit UI, communicating with the backend over HTTP only.
- `src/main.py`: FastAPI composition root.

The Docker image runs nginx on public port 7860, with FastAPI on loopback port 8000 and Streamlit on loopback port 8501 in one container. Only `/`, `/health`, and `/ready` are public; nginx returns 404 for `/query`, `/docs`, and `/openapi.json`, while Streamlit calls `/query` directly over loopback. This shape is a structural requirement of the Hugging Face Docker Space deploy target — see [`docs/adr/007-hf-pro-docker-space-deploy.md`](docs/adr/007-hf-pro-docker-space-deploy.md).

## Setup

Python 3.11 is the production and CI target.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

Copy `.env.example` to `.env`. `GROQ_API_KEY` is needed for primary generation; `OPENAI_API_KEY` is the optional fallback. If `API_KEY` is set, clients must send the same value as `X-API-Key` to `POST /query`. Never commit real credentials.

## Build the corpus and indexes

```bash
python -m src.features.ingestion.cli
python -m src.features.retrieval.cli
```

These commands write regenerated, gitignored artifacts to `ingestion/output/` and `retrieval/output/`. The lexical index is JSON (`retrieval/output/bm25_index.json`), never pickle. The first retrieval build downloads `BAAI/bge-m3`; after it is cached, `HF_HUB_OFFLINE=1` avoids network checks.

## Run the API and UI

```bash
uvicorn src.main:app --reload
```

The API exposes `GET /health`, `GET /ready`, and `POST /query` on port 8000. In a second terminal:

```bash
streamlit run src/web/app.py
```

The UI reads `API_BASE_URL` (default `http://localhost:8000`) and the optional `API_KEY` from its environment.

## Evaluation

```bash
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.retrieval_eval
python -m src.features.evaluation.threshold_analysis
python -m src.features.evaluation.generation_eval
```

Generation evaluation makes real third-party LLM calls, waits between questions for rate limits, and can incur API usage. It writes the automated report and a manual-review CSV; citation accuracy and faithfulness remain human-owned verdicts.

## Docker

```bash
docker build -t rag4 .
docker run --env-file .env -p 7860:7860 rag4
```

An earlier amd64 image was built and run end to end, but that result does not approve the hardened Dockerfile in the current deployment change. A fresh build, resource measurement, shutdown test, external route test, and bilingual query test remain required before promotion. amd64 is the real deploy target; the earlier ARM64 work remains only as historical record in `SPEC.md`.

**Hugging Face deployment:** create `JehuLara/manufacturing-rag-assistant-live` manually as a public Docker Space on CPU Basic, configure the repo-scoped Trusted Publisher and the `hf-live` GitHub Environment, then invoke `deploy-hf-space.yml` with a full 40-character green SHA and exact confirmation `DEPLOY`. The workflow stages an allowlist, writes deployment provenance, and intentionally mirrors it with deletion enabled. Runtime secrets belong only in HF Settings: `GROQ_API_KEY`, `OPENAI_API_KEY`, and `API_KEY`.

## Quality gates

```bash
ruff check src tests
mypy src
python -m src.features.evaluation.eval_set_integrity --verify
pytest
```

The current suite contains 168 passing tests. CI runs the same lint, type, integrity, and test gates on pushes and pull requests.

## Evidence

- [`eval/reports/retrieval_report_v1.0.0.md`](eval/reports/retrieval_report_v1.0.0.md)
- [`eval/reports/generation_eval_v1.0.0.md`](eval/reports/generation_eval_v1.0.0.md)
- [`eval/reports/manual_review_checklist_v1.0.0.csv`](eval/reports/manual_review_checklist_v1.0.0.csv)
- [`corpus/SOURCES.md`](corpus/SOURCES.md)
