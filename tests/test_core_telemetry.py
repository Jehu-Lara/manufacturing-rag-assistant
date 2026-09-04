"""Verifies the OTel span wiring named in ADR-006 actually exists at the
right call sites, without touching the process-global TracerProvider
(verified directly: opentelemetry.trace.set_tracer_provider silently
ignores a second call in the same process — "Overriding of current
TracerProvider is not allowed" — so tests can't reliably swap it per-test;
mocking get_tracer's return value is the reliable approach)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from src.adapters.secondary.embedder.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.adapters.secondary.llm.groq_openai_client import GroqOpenAiLlmClient
from src.core.config import Settings
from src.core.telemetry import configure_tracing, get_tracer
from src.domain.models import RetrievalResult
from src.features.query.use_cases import QueryUseCase
from src.features.retrieval.use_cases import HybridRetriever
from tests.fakes import InMemoryLLMClient, InMemoryRetriever


def _mock_tracer():
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=None)
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    return tracer


class _StubVectorStore:
    def build_collection(self, chunks, embedding_inputs):
        raise NotImplementedError

    def query(self, text, top_n):
        return [("chunk-1", 0.9, {"document_id": "doc-1"})]

    def get_metadata(self, chunk_id):
        raise NotImplementedError

    def ping(self):
        return True


class _StubLexicalIndex:
    def build_index(self, chunks, *, chunks_sha256):
        raise NotImplementedError

    def query(self, text, top_n):
        return []


def test_hybrid_retriever_creates_retrieval_hybrid_query_span():
    tracer = _mock_tracer()
    with patch("src.features.retrieval.use_cases.get_tracer", return_value=tracer):
        HybridRetriever(_StubVectorStore(), _StubLexicalIndex()).retrieve("q", k=1)
    tracer.start_as_current_span.assert_called_once_with("retrieval.hybrid.query")


def test_llm_adapter_creates_llm_generate_span():
    tracer = _mock_tracer()
    settings = Settings(
        groq_api_key="k", openai_api_key=None, llm_provider="groq", refusal_cosine_threshold=0.5, log_level="INFO"
    )
    with patch("src.adapters.secondary.llm.groq_openai_client.get_tracer", return_value=tracer):
        with patch.object(
            GroqOpenAiLlmClient,
            "_generate_structured_impl",
            return_value={"answer": "x", "citations": [], "refused": False},
        ):
            asyncio.run(
                GroqOpenAiLlmClient.from_settings(settings).generate_structured("sys", "user", {})
            )
    tracer.start_as_current_span.assert_called_once_with("llm.generate")


def test_query_use_case_creates_query_answer_question_span():
    tracer = _mock_tracer()
    settings = Settings(
        groq_api_key="k", openai_api_key=None, llm_provider="groq", refusal_cosine_threshold=0.5, log_level="INFO"
    )
    retriever = InMemoryRetriever(
        [
            RetrievalResult(
                chunk_id="c1",
                fused_score=1.0,
                semantic_rank=1,
                semantic_score=0.9,
                bm25_rank=1,
                bm25_score=1.0,
                metadata={
                    "document_id": "d",
                    "document_title": "t",
                    "section_heading": "s",
                    "revision": "r",
                    "source_type": "public",
                    "chunk_id": "c1",
                    "chunk_text": "text",
                },
            )
        ]
    )
    llm = InMemoryLLMClient(response={"answer": "x", "citations": [{"chunk_id": "c1"}], "refused": False})
    use_case = QueryUseCase(retriever, llm, settings)
    with patch("src.features.query.use_cases.get_tracer", return_value=tracer):
        asyncio.run(use_case.answer_question("q", "en"))
    tracer.start_as_current_span.assert_called_once_with("query.answer_question")


def test_embedder_creates_embedder_compute_span():
    tracer = _mock_tracer()
    embedder = SentenceTransformersEmbedder()
    fake_model = MagicMock()
    fake_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
    embedder._model = fake_model
    with patch("src.adapters.secondary.embedder.sentence_transformers_embedder.get_tracer", return_value=tracer):
        embedder.embed_texts(["hello"])
    tracer.start_as_current_span.assert_called_once_with("embedder.compute")


def test_configure_tracing_is_idempotent_and_does_not_raise():
    configure_tracing(app=None)  # first call in this process may already have run via other tests' imports
    configure_tracing(app=None)  # must not raise on a second call
    assert get_tracer() is not None


def test_telemetry_has_no_module_level_mutable_state():
    """CLAUDE.md's prohibited-patterns rule: no module-level mutable singleton.
    `_configured` was the last survivor of the pre-refactor _model/_CACHE/
    _encoding globals the src/ migration eliminated."""
    import ast
    from pathlib import Path

    module = Path(__file__).resolve().parent.parent / "src" / "core" / "telemetry.py"
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)], "telemetry uses `global`"
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_configured" not in assigned


def test_configure_tracing_installs_our_provider_and_reuses_it():
    from fastapi import FastAPI
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    configure_tracing(FastAPI())
    first = trace.get_tracer_provider()
    configure_tracing(FastAPI())

    assert trace.get_tracer_provider() is first
    assert isinstance(first, TracerProvider)
    assert first.resource.attributes.get("service.name") == "rag4-api"
