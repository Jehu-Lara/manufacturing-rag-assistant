# Manufacturing Knowledge RAG Assistant

Portfolio Project 4: a bilingual (EN/ES), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality procedures, with threshold-based refusal when retrieval does not support a confident answer.

See [`SPEC.md`](SPEC.md) for the complete scope, decisions, measured limitations, and data-honesty policy. See [`CLAUDE.md`](CLAUDE.md) for current contributor commands and architecture rules.

## Current status

- Corpus and ingestion: complete — 14 documents (9 public, 5 clearly labeled synthetic) produce 228 metadata-complete chunks.
- Retrieval: complete — hybrid BM25 + `BAAI/bge-m3` retrieval. On the profile actually served (`contextual-v1` / `expansion_mode=off`) against the frozen `eval_set` v1.1.0, Recall@5 is **0.887** overall — English 0.917 (n=48), Spanish 0.844 (n=32); Recall@3 0.825, MRR 0.721. Report: [`eval/reports/retrieval_report_v1.1.0__contextual-v1__off.md`](eval/reports/retrieval_report_v1.1.0__contextual-v1__off.md). The older headline 0.833 was `eval_set` v1.0.0 on the pre-`contextual-v1` index and is kept only in `SPEC.md` as history.
- Generation/API/UI: implemented and tested — FastAPI, Streamlit, Groq (`openai/gpt-oss-120b`) with OpenAI (`gpt-4o-mini`) fallback, bilingual responses, deterministic refusal gate, and citations resolved from retrieved metadata.
- Evaluation: the generation-side numbers — correct-refusal 0.900, false-refusal 0.200 (documented exception), citation accuracy 23/30 = 0.767 (below target), faithfulness 29/30 = 0.967 — are **historical measurements taken on `eval_set` v1.0.0 against the `raw-v1` index**. They have **not** been re-measured on the `contextual-v1` profile that ships today, so they describe an index the system no longer serves. The citation and faithfulness verdicts are human-reviewed, not LLM-as-judge scores. Re-running `generation_eval` makes real, paid third-party LLM calls and is owner-gated.
- Review-floor sweep (PR #6): complete — 125-question sweep across 24 candidate rules (floors 0.50–0.55 × lexical/semantic agreement signals) proved that 0 candidate rules cleared all global gates G1–G4 without unanswerable promotions or regressions. This established the pre-registered floor of `0.5500` as the optimal invariant ("diagnóstico sin cambio" / zero production code changes). Report: [`docs/eval/floor_sweep_summary.md`](docs/eval/floor_sweep_summary.md).
- Phase 3C generation pilot (PR #7): complete — single-repeat pilot execution on Groq `openai/gpt-oss-120b` across the frozen 48-question holdout and regression canaries demonstrated that `grounded_review` dropped false refusal from 33.3% (`binary`) down to 8.3%, with 0 unanswerable queries answered. The pilot was an exploratory single-repeat run affected by 429 rate limits; the pre-registered 3-repeat confirmation run and blind checklist grading remain required before any default policy flip from `binary` to `grounded_review`. Report: [`docs/eval/gate-generation-pilot-20260902.md`](docs/eval/gate-generation-pilot-20260902.md).
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
pip install --require-hashes -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
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
# Integrity verification of frozen evaluation and regression datasets
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.regression_set_integrity --verify
python -m src.features.evaluation.gate_holdout_integrity --verify

# Retrieval, threshold, and refusal score band guards
python -m src.features.evaluation.retrieval_eval
python -m src.features.evaluation.regression_eval
python -m src.features.evaluation.threshold_analysis
python -m src.features.evaluation.gate_score_guard
python -m src.features.evaluation.gate_holdout_profile

# Review-floor calibration sweep across candidate thresholds and agreement signals
python -m src.features.evaluation.floor_sweep

# Generation evaluation (makes real third-party LLM calls; owner-gated)
python -m src.features.evaluation.generation_eval
python -m src.features.evaluation.gate_generation_eval --provider {groq|openai}
python -m src.features.evaluation.gate_generation_eval --import-verdicts <run_dir>
```

Generation evaluation makes real third-party LLM calls, waits between questions for rate limits, and can incur API usage. It writes the automated report and a manual-review CSV; citation accuracy and faithfulness remain human-owned verdicts. The causal A/B runner `gate_generation_eval` replays retrieval snapshots across `binary` and `grounded_review` policy arms, verifies checksums, and requires `--import-verdicts` to incorporate blind human review without tampering.

## Docker

```bash
docker build -t rag4 .
docker run --env-file .env -p 7860:7860 rag4
```

An earlier amd64 image was built and run end to end, but that result does not approve the hardened Dockerfile in the current deployment change. A fresh build, resource measurement, shutdown test, external route test, and bilingual query test remain required before promotion. amd64 is the real deploy target; the earlier ARM64 work remains only as historical record in `SPEC.md`.

**Hugging Face deployment:** create `JehuLara/manufacturing-rag-assistant-live` manually as a public Docker Space on CPU Basic, configure the repo-scoped Trusted Publisher and the `hf-live` GitHub Environment, then invoke `deploy-hf-space.yml` with a full 40-character green SHA and exact confirmation `DEPLOY`. The workflow stages an allowlist, rejects nested secret-like paths and symlinks, scans the final staging tree with a checksum-pinned Gitleaks binary, writes deployment provenance, and intentionally mirrors it with deletion enabled. Runtime secrets belong only in HF Settings: `GROQ_API_KEY`, `OPENAI_API_KEY`, and `API_KEY`. Questions and retrieved context are sent to the selected external LLM provider, so the UI warns users not to submit confidential, personal, regulated, or proprietary information.

## Quality gates

```bash
ruff check src tests
mypy src
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.regression_set_integrity --verify
python -m src.features.ingestion.cli
python -m src.features.retrieval.cli
python -m src.features.evaluation.gate_score_guard
pytest
python -m src.features.evaluation.gate_holdout_integrity --verify
```

For the current test count, run `pytest --collect-only -q` — a number pinned in prose goes stale silently and then reads as a claim the repo no longer supports. CI runs these exact lint, type, dataset integrity, corpus build, score guard, test suite, and holdout gates on pushes and pull requests (`.github/workflows/ci.yml`).

## Evidence

- [`eval/reports/retrieval_report_v1.1.0__contextual-v1__off.md`](eval/reports/retrieval_report_v1.1.0__contextual-v1__off.md) — the served profile
- [`eval/reports/retrieval_report_v1.0.0.md`](eval/reports/retrieval_report_v1.0.0.md) — historical (`eval_set` v1.0.0)
- [`eval/reports/generation_eval_v1.0.0.md`](eval/reports/generation_eval_v1.0.0.md)
- [`eval/reports/manual_review_checklist_v1.0.0.csv`](eval/reports/manual_review_checklist_v1.0.0.csv)
- [`eval/reports/regression_eval_v1.1.0__contextual-v1__off.md`](eval/reports/regression_eval_v1.1.0__contextual-v1__off.md) — regression evaluation on served profile
- [`docs/eval/gate-generation-pilot-20260902.md`](docs/eval/gate-generation-pilot-20260902.md) — Phase 3C LLM generation pilot report (Groq `openai/gpt-oss-120b`)
- [`docs/eval/floor_sweep_summary.md`](docs/eval/floor_sweep_summary.md) — 125-sweep review-floor calibration summary
- [`tests/test_adversarial_challenger.py`](tests/test_adversarial_challenger.py) — 14-test adversarial verification suite (Score Inversion Theorem proof)
- [`corpus/SOURCES.md`](corpus/SOURCES.md)
