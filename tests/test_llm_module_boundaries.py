from __future__ import annotations

import ast
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parent.parent / "src" / "adapters" / "secondary" / "llm"


def _imports(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _top_level(module: Path) -> set[str]:
    return {m.split(".")[0] for m in _imports(module)}


def test_validation_is_sdk_free() -> None:
    """Schema validation and repair-prompt construction are pure text work.
    Keeping the SDKs out of this module is what lets it be tested without a
    single mock."""
    assert not ({"groq", "openai"} & _top_level(LLM_DIR / "validation.py"))


def test_tracing_is_sdk_free_and_carries_no_prompt_text() -> None:
    source = (LLM_DIR / "tracing.py").read_text(encoding="utf-8")
    assert not ({"groq", "openai"} & _top_level(LLM_DIR / "tracing.py"))
    for banned in ("system_prompt", "user_prompt", "api_key"):
        assert banned not in source, f"tracing.py mentions {banned!r} — trace events are content-free"


def test_only_transport_imports_the_provider_sdks() -> None:
    """The SDK boundary is one module, so the test suite has exactly one place
    to patch (CLAUDE.md's mocking convention)."""
    offenders = {
        module.name
        for module in LLM_DIR.glob("*.py")
        if module.name != "transport.py" and ({"groq", "openai"} & _top_level(module))
    }
    assert not offenders, f"provider SDKs imported outside transport.py: {offenders}"


def test_public_import_paths_survive_the_split() -> None:
    """The façade is the compatibility contract: app.lifespan, the Phase 3C
    runner and the test suite all import these from groq_openai_client."""
    from src.adapters.secondary.llm.groq_openai_client import (  # noqa: F401
        GROQ_MODEL,
        MAX_COMPLETION_TOKENS,
        OPENAI_MODEL,
        RATE_LIMIT_BACKOFF_SECONDS,
        GroqOpenAiLlmClient,
        LlmTraceEvent,
        TraceHook,
        log_llm_trace,
    )


def test_trace_events_never_carry_a_key_or_prompt_text() -> None:
    """LlmTraceEvent reaches production stdout through log_llm_trace, so its
    field list is a disclosure boundary, not an internal record."""
    from dataclasses import fields

    from src.adapters.secondary.llm.tracing import LlmTraceEvent

    names = {f.name for f in fields(LlmTraceEvent)}
    assert not (
        names & {"system_prompt", "user_prompt", "question", "answer", "api_key", "detail", "message"}
    )


def test_provider_error_surface_carries_type_and_status_only() -> None:
    """A provider's error body can echo prompt fragments back, so str(exc) is
    deliberately absent from the trace."""
    from src.adapters.secondary.llm.tracing import _provider_error_trace_fields

    exc = RuntimeError("leaked: the secret question text")
    fields_out = _provider_error_trace_fields(exc)

    assert "leaked" not in repr(fields_out)
    assert fields_out == {"error_type": "RuntimeError"}


def test_aclose_is_idempotent_and_closes_nothing_when_unused() -> None:
    import asyncio

    from src.adapters.secondary.llm.groq_openai_client import GroqOpenAiLlmClient

    client = GroqOpenAiLlmClient(provider="groq")
    asyncio.run(client.aclose())
    asyncio.run(client.aclose())
