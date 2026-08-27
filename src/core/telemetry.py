from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def configure_tracing(app: "FastAPI") -> None:
    """No-op in Phase 1 — wired to real OTel FastAPI instrumentation and
    retrieve/embed/generate spans in Phase 2 (see ADR-006)."""
    return None
