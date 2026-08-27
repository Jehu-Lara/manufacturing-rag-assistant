# ADR-005: Deploy/container shape — current reality vs. deferred split

## Context

**Correction (recorded after this ADR's first version): the deploy target
was not Hugging Face Docker Spaces at the time this refactor happened.**
Per `SPEC.md`'s Phase 3 status, HF Spaces was dropped as the deploy target
on 2026-08-25 — HF's free tier no longer permits *running* a Docker SDK
Space at all (PRO, ~$9/month, required, with no free workaround), and the
project owner declined to pay for it. The real target since that pivot is
**Oracle Cloud's "Always Free" Ampere A1 (ARM) VM** — a normal, fully
user-controlled VM, not a constrained one-port PaaS. A free HF *Static*
Space (`docs/hf-space/`, no compute, no PRO needed) remains planned only as
a recruiter-visibility showcase page linking to the live Oracle-hosted demo
— it never runs this project's container.

This ADR's original version reasoned from the HF-Spaces-era constraints
(one exposed port, no persistent volume at the free tier — hence baking
the vector/BM25 indexes into the image at build time) as if they still
applied. They don't: Oracle Cloud Always Free is a real VM with normal
multi-port networking and persistent disk. The current Dockerfile still
bakes `uvicorn` (FastAPI, :8000), `streamlit` (:8501), and `nginx` (reverse
proxy, :7860) into one container — but that shape is now a **carried-over
default from the abandoned HF-Spaces target, not something the actual
current target (Oracle Cloud) requires.**

## Decision (current, real — this ADR does not decide the deferred part)

Keep the single-container shape as-is for *this refactor*, verified
working end-to-end via a real `docker build`/`docker run` (both `/health`
and the new `/ready` correctly proxied through nginx, `index_loaded: true`
against the real baked-in index). The code-level separation (`src/web/`
vs. the API's `src/adapters/primary/http/` + `src/features/*`) is already
real and enforced by the import-invariant test regardless of how the two
processes are actually packaged and deployed. What ISN'T decided here is
whether to actually split the container.

## Consequences / explicitly deferred

The `docker-compose.yml` split, the nginx/`start.sh` deletion, and moving
index population out of the Docker build are **deferred** (see the
migration plan's Phase 4b appendix) — but unlike this ADR's original
reasoning, the blocker is **not** a deploy-target constraint (Oracle Cloud
supports multi-service compose and persistent volumes with no HF-Spaces-
style one-port limitation). The real blocker is that **Oracle Cloud
Always Free ARM provisioning itself is stalled** on Oracle's own regional
capacity limit (`SPEC.md`: "Out of host capacity" in the Monterrey region,
2026-08-26) — native ARM build/run verification has only happened under
QEMU emulation so far (~2 hours of clean, error-free embedding computation
before being deliberately stopped, not run to completion), not on real
hardware. Splitting the container now, before a VM actually exists to test
against, would be designing blind for an environment nobody has run code
on yet.

**Revised recommendation, once Oracle capacity frees up**: re-open this
decision. Given the real target is a full VM with no port/volume
constraint, the target-tree's original `docker-compose.yml` split (indexes
as a mounted volume populated by the ingestion/retrieval CLIs, not baked
into the image; API container running only `uvicorn src.main:app`, no
nginx, no streamlit) is likely the *more* appropriate shape for Oracle
Cloud, not a HF-Spaces-only nicety being needlessly deferred. This ADR
does not make that call — it only corrects the reasoning and flags it for
the human to decide once a VM is actually provisionable.

Docker CLI v29.7.2 with the compose v5.4.0 plugin are confirmed available
and working in the current dev environment (a stale note elsewhere in this
repo's `CLAUDE.md` claiming Docker Desktop's WSL2 backend was unavailable
was outdated — corrected separately).
