from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.models import ExpansionMode, IndexProfile
from src.features.retrieval import index_manifest
from src.features.retrieval.chunk_store import load_chunks
from src.features.retrieval.use_cases import HybridRetriever


def build_retriever(
    expansion_mode: ExpansionMode = "off",
    *,
    expected_profile: IndexProfile | None = None,
    verify_physical_coherence: bool = True,
) -> HybridRetriever:
    """`verify_physical_coherence` runs the same manifest + Chroma + BM25 cross
    check as the serving process (`app.lifespan`) before the retriever is
    returned, so no eval runner can measure an index whose manifest, vector
    collection and lexical index physically disagree on profile, count or
    model. Disabled only where the caller has already run the check."""
    settings = load_settings()
    profile: IndexProfile = expected_profile or index_manifest.resolve_index_profile(settings)

    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_path, embedder=embedder, index_profile=profile
    )
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)

    if verify_physical_coherence:
        manifest = index_manifest.verify(expected_profile=profile)
        chunk_ids = [chunk.chunk_id for chunk in load_chunks()]
        vector_store.validate_collection(
            expected_profile=profile, expected_count=manifest.chunk_count
        )
        # Identical to the cross-check src.adapters.primary.http.app.lifespan
        # runs, deliberately: no eval runner may measure an index whose three
        # artifacts physically disagree on profile, count, model OR content.
        lexical_index.validate(
            chunk_ids,
            expected_chunks_sha256=manifest.chunks_sha256,
            expected_lexical_profile=manifest.lexical_profile,
        )

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
    index_manifest.verify()
