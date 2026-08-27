# ADR-005: Deploy/container shape — current reality vs. deferred split

## Context

The current Dockerfile bakes `uvicorn` (FastAPI, :8000), `streamlit`
(:8501), and `nginx` (reverse proxy, :7860) into a single container, because
the deploy target (a Hugging Face Docker Space) requires exactly one
exposed port and there is no persistent volume at the free tier — so the
vector/BM25 indexes are also baked into the image at build time, not
populated at container start.

The blueprint's target architecture (one API runtime, `src/web/` as a
separate, HTTP-only Streamlit process) implies a `docker-compose.yml`
`api` + `web` service split, with the API container running only
`uvicorn src.main:app` — no nginx, no streamlit, no build-time index baking.

## Decision (current, real — this ADR does not decide the deferred part)

Keep the single-container shape as-is for this refactor. The code-level
separation (`src/web/` vs. the API's `src/adapters/primary/http/` +
`src/features/*`) is already real and enforced by the import-invariant test
— `src/web/` has zero imports of `src.domain/features/adapters` regardless
of how the two processes are actually packaged and deployed. What ISN'T
decided here is whether to actually split the container.

## Consequences / explicitly deferred

The `docker-compose.yml` split, the nginx/`start.sh` deletion, and moving
index population out of the Docker build are **deferred** (see the
migration plan's Phase 4b appendix), blocked on a separate decision: does
this project keep targeting HF Docker Spaces' one-container/one-port
constraint, or move to a host that supports multi-service compose (which
would also resolve the free-tier no-persistent-volume constraint that
motivates baking indexes into the image today)? Implementing the literal
target-tree split without first answering that question would break the
live, working deploy for no immediate benefit.

Docker CLI v29.7.2 with the compose v5.4.0 plugin are confirmed available
in the current dev environment (a stale note elsewhere in this repo's
`CLAUDE.md`, claiming Docker Desktop's WSL2 backend was unavailable,
appears outdated — worth correcting separately, but doesn't change this
ADR's decision).
