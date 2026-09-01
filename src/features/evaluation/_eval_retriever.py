from __future__ import annotations

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.lexical.bm25_lexical_index import Bm25LexicalIndex
from src.adapters.secondary.vector.chroma_vector_store import ChromaVectorStore
from src.core.config import load_settings
from src.domain.models import ExpansionMode
from src.features.retrieval.use_cases import HybridRetriever


def build_retriever(expansion_mode: ExpansionMode = "off") -> HybridRetriever:
    settings = load_settings()
    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(persist_dir=settings.chroma_path, embedder=embedder)
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)
    return HybridRetriever(vector_store, lexical_index, expansion_mode=expansion_mode)
