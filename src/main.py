from __future__ import annotations

from src.adapters.primary.http.app import create_app

# Deliberately lightweight: builds the FastAPI object + mounts the router
# only. Adapter construction (model load, Chroma connection, BM25 load)
# happens inside create_app's lifespan, not here at import time — importing
# this module (or running `uvicorn src.main:app`) must never trigger a real
# model load until the app actually starts serving.
app = create_app()

