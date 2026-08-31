from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.models import ExpansionMode, IndexProfile
from src.features.retrieval import index_manifest
from src.features.retrieval.use_cases import HybridRetriever


def build_retriever(expansion_mode: ExpansionMode = "off") -> HybridRetriever:
    settings = load_settings()
    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)
    return HybridRetriever(vector_store, lexical_index, expansion_mode=expansion_mode)


def assert_live_index_profile(expected: IndexProfile) -> None:
    """Raise if the live index manifest was not built with `expected`.
    Guards against measuring a stale index and labelling the report wrongly."""
    try:
        manifest = index_manifest.read()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"no index manifest at {index_manifest.MANIFEST_FILE} — run "
            f"`INDEX_PROFILE={expected} python -m src.features.retrieval.cli` first"
        ) from exc
    if manifest.index_profile != expected:
        raise RuntimeError(
            f"live index is {manifest.index_profile!r} but this run is labelled "
            f"{expected!r}; rebuild with INDEX_PROFILE={expected} or pass the "
            f"matching index_profile"
        )
