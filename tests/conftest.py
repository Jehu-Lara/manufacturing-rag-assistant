from __future__ import annotations

from src.core.config import load_settings
from src.features.retrieval import index_manifest


def built_retrieval_index_present() -> bool:
    """True when a real retrieval index is on disk (manifest + Chroma dir +
    BM25 file). Tests that start the FastAPI app trigger lifespan, which now
    hard-validates the index — on a fresh clone with no `python -m
    src.features.retrieval.cli` run yet, those tests skip instead of erroring.
    CI always builds the index first, so it runs them for real."""
    settings = load_settings()
    return (
        index_manifest.MANIFEST_FILE.exists()
        and settings.chroma_path.exists()
        and settings.bm25_path.exists()
    )


REQUIRES_BUILT_INDEX_REASON = (
    "retrieval index not on disk — run `python -m src.features.retrieval.cli` "
    "(CI builds it before the test step)"
)
