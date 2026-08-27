from __future__ import annotations

# Composition root — wired for real in Phase 2 (src.adapters.primary.http.app.create_app,
# with adapter construction happening inside FastAPI's lifespan, not at import time here).
