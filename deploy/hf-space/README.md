---
title: Manufacturing RAG Assistant Live
emoji: 🏭
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: mit
suggested_hardware: cpu-basic
models:
  - BAAI/bge-m3
tags:
  - rag
  - manufacturing
  - fastapi
  - streamlit
  - multilingual
---

# Manufacturing RAG Assistant Live

Interactive English/Spanish retrieval-augmented assistant over a bounded
manufacturing corpus. The public UI is served at `/`; `/health` and `/ready`
are the only public service endpoints. The query API remains internal to the
container and requires an API key.

Deployment is manual from a selected green commit in
`Jehu-Lara/manufacturing-rag-assistant`. The deployed source revision is
recorded in `DEPLOYED_SHA` and `DEPLOYED_AT` at the Space repository root.

The Space uses CPU Basic. Cold starts, the 16 GB memory ceiling, ephemeral
storage, and availability of the configured external LLM provider remain
operational constraints.
