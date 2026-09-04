# Bucket 2 — `LLMClientPort` Decoupling and LLM Adapter Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take `Settings` out of `LLMClientPort.generate_structured`, injecting provider and API keys (as `SecretStr`) at construction instead, and split the 634-line `groq_openai_client.py` into transport / validation / tracing / failover modules — preserving content-free traces, single-owner teardown, and safe error surfaces exactly as they behave today.

**Architecture:** Three tasks, strictly ordered. T1 changes the port signature and the client constructor, and mechanically re-points every one of the ~50 call sites; it is the only task with a behaviour change (per-call provider/key switching becomes per-instance). T2 splits the module into four files behind unchanged public import paths, which also moves the SDK patch targets the test suite relies on. T3 re-points those patch targets, adds the secret-non-disclosure and teardown contract tests, and syncs the docs. T1 → T2 → T3.

**Tech Stack:** Python 3.11, groq 1.7.0 / openai 3.x async SDKs, pydantic `SecretStr`, pytest + `unittest.mock.AsyncMock`, opentelemetry spans.

**Spec:** `docs/superpowers/specs/2026-09-04-architecture-remediation-design.md`

## Global Constraints

- Execution is gated: PR #9 resolved by the owner, then a new branch cut from `master` with the owner's authorization. No commits, pushes, merges or deploys without a separate explicit request.
- **No real provider call in any test.** The SDK boundary stays mocked (`AsyncMock`) and higher layers keep using port fakes — CLAUDE.md's scoped mocking exception, unchanged.
- `LlmTraceEvent`'s field list is a disclosure boundary: it reaches production stdout through `log_llm_trace`. Do not add a field, and never let a prompt, question, answer, API key or raw exception string into one.
- Citation integrity is untouched: `chunk_id` is still the only LLM-supplied citation field trusted; `CitationResolver` and `GroundedEvidenceResolver` stay fail-closed.
- Byte-stable invariants unchanged (`0.5999`, `0.5500`, RRF `k=60`, `binary`, `off`, `contextual-v1`).
- After each task: `pytest tests/test_llm_adapter.py tests/test_query_use_case.py -q` green. End of bucket: `pytest -q`, `ruff check src tests`, `mypy src` green.

---

## File Structure

- `src/domain/ports.py` — `LLMClientPort.generate_structured(system_prompt, user_prompt, schema)`. The `from src.core.config import Settings` import disappears, so `src/domain/` stops depending on `src/core/config` entirely.
- `src/adapters/secondary/llm/tracing.py` — **new.** `LlmTraceEvent`, `TraceHook`, `log_llm_trace`, `_provider_error_fields`, `_provider_error_trace_fields`, `_usage_trace_fields`. No SDK orchestration, no prompt text.
- `src/adapters/secondary/llm/validation.py` — **new.** `_try_parse_and_validate`, `_validate_against_schema`, `_build_repair_system_prompt`. Pure functions; no `groq`/`openai` import.
- `src/adapters/secondary/llm/transport.py` — **new.** Model/token constants, `_messages`, `_is_unsupported_response_format_error`, `_extract_retry_after_seconds`, the error-type tuples, the provider-keyed SDK client cache (`_sdk_client`, `aclose`), `_invoke`, `_call_groq`, `_call_openai`. **This is the module `groq` and `openai` are imported into**, so it becomes the patch target for every SDK mock.
- `src/adapters/secondary/llm/groq_openai_client.py` — keeps only `GroqOpenAiLlmClient`: constructor injection, provider failover, rate-limit backoff, JSON-repair orchestration, the `llm.generate` span. Re-exports the public names (`LlmTraceEvent`, `TraceHook`, `log_llm_trace`, `GROQ_MODEL`, `OPENAI_MODEL`, `RATE_LIMIT_BACKOFF_SECONDS`) so no import path outside this package changes.
- `src/features/query/use_cases.py:210` — drops the `settings` argument.
- `src/features/evaluation/gate_generation_eval.py:128-136`, `:954` and `src/features/evaluation/generation_eval.py:54` — updated call and construction sites.
- `src/adapters/primary/http/app.py:54` — constructs from settings.
- `tests/fakes.py:32`, `tests/test_llm_adapter.py`, `tests/test_llm_client_json_repair.py`, `tests/test_query_use_case.py:291-294`, `tests/test_evaluation_gate_generation_eval.py:65,115-122,771`, `tests/test_evaluation_generation_eval.py:184,353-356`, `tests/test_core_telemetry.py:67-70` — signature updates.

---

### Task 1: Constructor injection — `Settings` leaves the port

**Files:**
- Modify: `src/domain/ports.py:36-40`, `src/adapters/secondary/llm/groq_openai_client.py:253-270,369-380,489+`, `src/features/query/use_cases.py:210`, `src/adapters/primary/http/app.py:54`, `src/features/evaluation/gate_generation_eval.py:128-136,954`, `src/features/evaluation/generation_eval.py:54`, `tests/fakes.py:32`
- Test: `tests/test_llm_adapter.py`, `tests/test_query_use_case.py`, `tests/test_domain_ports.py` (create if absent)

**Interfaces:**
- Produces:
  - `LLMClientPort.generate_structured(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]`
  - `GroqOpenAiLlmClient(*, provider: LlmProvider, groq_api_key: SecretStr | None = None, openai_api_key: SecretStr | None = None, allow_provider_fallback: bool = True, trace_hook: TraceHook | None = None, rate_limit_backoff_seconds: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS)`
  - `GroqOpenAiLlmClient.from_settings(settings: Settings, **overrides: Any) -> GroqOpenAiLlmClient` — the composition-root convenience; `overrides` forwards the three non-credential keywords.
- Consumes: `src.core.config.{LlmProvider, Settings}` — now imported by the **adapter**, which is a layer allowed to know about config, instead of by `src/domain/ports.py`, which is not.

**Behaviour change to state explicitly:** provider and credentials become per-instance instead of per-call. Rotating a key or switching provider means constructing a new client. Serving already passes one stable `Settings` for the process lifetime and the offline runners are sequential, so nothing in the repo relied on per-call switching — but two existing tests did, and Step 4 rewrites them rather than deleting them.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_llm_adapter.py`:

```python
def test_generate_structured_takes_no_settings_argument() -> None:
    """The port is a domain type; a Settings parameter drags src/core/config
    into every implementation and every fake. Provider and credentials are
    construction-time facts, not per-call ones."""
    import inspect

    from src.domain.ports import LLMClientPort

    params = list(inspect.signature(LLMClientPort.generate_structured).parameters)
    assert params == ["self", "system_prompt", "user_prompt", "schema"]


def test_domain_ports_does_not_import_core_config() -> None:
    import ast
    from pathlib import Path

    module = Path(__file__).resolve().parent.parent / "src" / "domain" / "ports.py"
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "src.core.config" not in imported


def test_from_settings_carries_provider_and_both_keys() -> None:
    from pydantic import SecretStr

    settings = _settings("openai")
    client = GroqOpenAiLlmClient.from_settings(settings, allow_provider_fallback=False)

    assert client._provider == "openai"
    assert isinstance(client._openai_api_key, (SecretStr, type(None)))
    assert client._allow_provider_fallback is False


def test_client_repr_never_leaks_a_key() -> None:
    from pydantic import SecretStr

    client = GroqOpenAiLlmClient(provider="groq", groq_api_key=SecretStr("sk-super-secret"))

    assert "sk-super-secret" not in repr(client)
    assert "sk-super-secret" not in str(vars(client))
```

The last test relies on `SecretStr` being stored unwrapped — that is the point of the assertion, and it is what makes `repr` safe by construction.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_llm_adapter.py -q -k "no_settings or core_config or from_settings or repr"`

Expected: FAIL — the port still declares `settings`, `src/domain/ports.py:5` still imports `Settings`, and `from_settings` does not exist.

- [ ] **Step 3: Change the port and the client**

`src/domain/ports.py` — delete the `from src.core.config import Settings` line and:

```python
@runtime_checkable
class LLMClientPort(Protocol):
    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...
```

`src/adapters/secondary/llm/groq_openai_client.py`:

```python
    def __init__(
        self,
        *,
        provider: LlmProvider,
        groq_api_key: Optional[SecretStr] = None,
        openai_api_key: Optional[SecretStr] = None,
        allow_provider_fallback: bool = True,
        trace_hook: Optional[TraceHook] = None,
        rate_limit_backoff_seconds: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        """Provider and credentials are construction-time facts. Keys are held
        as SecretStr and unwrapped only at the SDK boundary in _api_key_for, so
        neither repr(), vars(), a traceback, nor a trace event can carry one.
        `rate_limit_backoff_seconds=()` means fail-fast: one physical attempt
        per provider, no sleep — serving wires that (a user-facing query must
        not sleep 105s behind nginx/httpx 60s timeouts) while keeping provider
        fallback; offline evaluation keeps the default long schedule."""
        self._provider: LlmProvider = provider
        self._groq_api_key = groq_api_key
        self._openai_api_key = openai_api_key
        self._allow_provider_fallback = allow_provider_fallback
        self._trace_hook = trace_hook
        self._rate_limit_backoff_seconds = tuple(rate_limit_backoff_seconds)
        self._clients: dict[str, tuple[Optional[str], Any]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "GroqOpenAiLlmClient":
        return cls(
            provider=settings.llm_provider,
            groq_api_key=settings.groq_api_key,
            openai_api_key=settings.openai_api_key,
            **overrides,
        )

    def _api_key_for(self, provider: str) -> Optional[str]:
        """The single unwrap point. Everything above this line handles SecretStr."""
        secret = self._groq_api_key if provider == "groq" else self._openai_api_key
        return secret.get_secret_value() if secret is not None else None
```

Delete the module-level `_api_key_for(provider, settings)` (line 112). Then in `generate_structured` / `_generate_structured_impl`, drop the `settings` parameter and replace `primary = settings.llm_provider` with `primary = self._provider` and `api_key = _api_key_for(provider, settings)` with `api_key = self._api_key_for(provider)`. `_get_provider_response`, `_call_provider`, `_call_groq`, `_call_openai` already take `api_key: Optional[str]` and need no change.

Keep the existing provider-keyed fingerprint cache in `_sdk_client` unchanged. With immutable per-instance keys its rotation branch is now unreachable in practice, but it costs nothing and still guards against a caller mutating `_groq_api_key` in place. Do not delete it in this task.

- [ ] **Step 4: Re-point every call and construction site**

Production, four sites:

```python
# src/adapters/primary/http/app.py:54
llm_client = GroqOpenAiLlmClient.from_settings(
    settings, trace_hook=log_llm_trace, rate_limit_backoff_seconds=()
)

# src/features/evaluation/gate_generation_eval.py:954
return GroqOpenAiLlmClient.from_settings(settings, allow_provider_fallback=False, trace_hook=hook)

# src/features/evaluation/generation_eval.py:54
llm_client = GroqOpenAiLlmClient.from_settings(settings)

# src/features/query/use_cases.py:210 — drop the trailing `self._settings` argument
llm_result = await self._llm_client.generate_structured(system_prompt, user_prompt, schema)
```

`gate_generation_eval.py:954` sits inside a factory that must already have a `settings` in scope — if it does not, thread the one `run()` loads rather than calling `load_settings()` a second time.

`gate_generation_eval.WithinRepeatCache.generate_structured` (lines 128–136) drops its `settings` parameter and forwards three arguments. Its cache key is `(system_prompt, user_prompt, _schema_key(schema))` and does not change — the confident-band call stays byte-identical between policies within a repeat, which is what makes the A/B causal.

`tests/fakes.py:32` — `InMemoryLLMClient.generate_structured(self, system_prompt, user_prompt, schema)`.

The test suite, mechanically:

```bash
grep -rn "generate_structured(" tests/ | grep -v "def generate_structured"
```

- Calls of the form `.generate_structured("system", "user", JSON_SCHEMA, _settings("groq"))` become `.generate_structured("system", "user", JSON_SCHEMA)`, and the construction on the same line becomes `GroqOpenAiLlmClient(provider="groq", groq_api_key=SecretStr("gk"), openai_api_key=SecretStr("ok"))`. Add a module-level helper to `tests/test_llm_adapter.py` so this stays one edit per line:

```python
def _client(provider: str = "groq", **overrides: Any) -> GroqOpenAiLlmClient:
    return GroqOpenAiLlmClient.from_settings(_settings(provider), **overrides)
```

- The "missing key" tests at `tests/test_llm_adapter.py:241` and `:254` currently build a `settings` with a key absent. They become `_client()` built from that same `_settings(...)`, so `from_settings` carries the `None` through — the assertion (provider skipped, `attempts_summary` records "not configured") is unchanged.
- Overridden fakes that call `super().generate_structured(...)` (`tests/test_query_use_case.py:291-294`, `tests/test_evaluation_generation_eval.py:353-356`) drop the fourth argument in both the signature and the `super()` call.
- `tests/test_core_telemetry.py:67-70` patches `_generate_structured_impl`; drop the `settings` argument from the `generate_structured` call there.

Two tests assert the removed behaviour and must be **rewritten, not deleted**:

- `tests/test_llm_adapter.py:639-641` (rotated key within one client) becomes: build one client with key A, make a call, then build a **second** client with key B and assert it constructs its own SDK client — plus an assertion that the first client's key is unreachable from the second.
- `tests/test_llm_adapter.py:662-664` (provider switch within one client) becomes two clients, `provider="groq"` and `provider="openai"`, each caching its own SDK client.

Add a comment on both explaining that per-call switching was removed deliberately in this bucket.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_llm_adapter.py tests/test_query_use_case.py tests/test_llm_client_json_repair.py -q
pytest -q
```

Expected: PASS. `mypy src` must also be green here — the `Protocol` change is exactly the kind of thing `--strict` catches at every implementation site.

---

### Task 2: Split the adapter into transport / validation / tracing / failover

**Files:**
- Create: `src/adapters/secondary/llm/tracing.py`, `src/adapters/secondary/llm/validation.py`, `src/adapters/secondary/llm/transport.py`
- Modify: `src/adapters/secondary/llm/groq_openai_client.py` (reduced to the orchestrator plus re-exports)
- Test: `tests/test_llm_module_boundaries.py` (new)

**Interfaces:**
- Produces:
  - `tracing.LlmTraceEvent`, `tracing.TraceHook`, `tracing.log_llm_trace`, `tracing.provider_error_fields(provider, exc)`, `tracing.provider_error_trace_fields(exc)`, `tracing.usage_trace_fields(response, model, schema_mode)`
  - `validation.try_parse_and_validate(raw_text, schema)`, `validation.validate_against_schema(instance, schema)`, `validation.build_repair_system_prompt(original_system_prompt, previous_response, error)`
  - `transport.ProviderTransport` — a class owning `_clients`, `_lock`, `sdk_client(provider, api_key)`, `aclose()`, `invoke(provider, phase, model, schema_mode, create)`, `call_groq(...)`, `call_openai(...)`, plus `GROQ_MODEL`, `OPENAI_MODEL`, `MAX_COMPLETION_TOKENS`, `RATE_LIMIT_ERROR_TYPES`, `JSON_SCHEMA_RETRY_ERROR_TYPES`, `extract_retry_after_seconds`, `messages`, `is_unsupported_response_format_error`
  - `groq_openai_client` re-exports `LlmTraceEvent`, `TraceHook`, `log_llm_trace`, `GROQ_MODEL`, `OPENAI_MODEL`, `RATE_LIMIT_BACKOFF_SECONDS` so `src/adapters/primary/http/app.py:14` and `src/features/evaluation/gate_generation_eval.py` keep their current import lines verbatim.
- `GroqOpenAiLlmClient.aclose()` delegates to `ProviderTransport.aclose()`, keeping today's idempotence and `ExceptionGroup` aggregation. Ownership is unchanged: exactly one owner closes exactly once — `app.lifespan`'s `finally`, or the offline runner's owning `asyncio.run`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_llm_module_boundaries.py`:

```python
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


def test_validation_is_sdk_free() -> None:
    """Schema validation and repair-prompt construction are pure text work.
    Keeping the SDKs out of this module is what lets it be tested without a
    single mock."""
    assert not ({"groq", "openai"} & {m.split(".")[0] for m in _imports(LLM_DIR / "validation.py")})


def test_tracing_is_sdk_free_and_carries_no_prompt_text() -> None:
    source = (LLM_DIR / "tracing.py").read_text(encoding="utf-8")
    assert not ({"groq", "openai"} & {m.split(".")[0] for m in _imports(LLM_DIR / "tracing.py")})
    for banned in ("system_prompt", "user_prompt", "question", "answer", "api_key"):
        assert banned not in source, f"tracing.py mentions {banned!r} — trace events are content-free"


def test_only_transport_imports_the_provider_sdks() -> None:
    """The SDK boundary is one module, so the test suite has exactly one place
    to patch (CLAUDE.md's mocking convention)."""
    offenders = {
        module.name
        for module in LLM_DIR.glob("*.py")
        if module.name != "transport.py"
        and {"groq", "openai"} & {m.split(".")[0] for m in _imports(module)}
    }
    assert not offenders, f"provider SDKs imported outside transport.py: {offenders}"


def test_public_import_paths_survive_the_split() -> None:
    from src.adapters.secondary.llm.groq_openai_client import (  # noqa: F401
        GROQ_MODEL,
        OPENAI_MODEL,
        RATE_LIMIT_BACKOFF_SECONDS,
        GroqOpenAiLlmClient,
        LlmTraceEvent,
        TraceHook,
        log_llm_trace,
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_llm_module_boundaries.py -q`

Expected: FAIL — `validation.py`, `tracing.py` and `transport.py` do not exist, and `groq_openai_client.py` imports both SDKs.

- [ ] **Step 3: Extract tracing and validation first (no behaviour change)**

Move, verbatim, from `groq_openai_client.py`:

- into `tracing.py`: the `LlmTraceEvent` dataclass and its docstring (lines 22–57), `log_llm_trace` (59–80), the `TraceHook` alias, `_provider_error_fields` (117–125), `_provider_error_trace_fields` (126–136), `_usage_trace_fields` (137–155). Drop the leading underscore on the three helpers as they are now cross-module.
- into `validation.py`: `_try_parse_and_validate` (186–199), `_validate_against_schema` (200–236), `_build_repair_system_prompt` (237–246). Same de-underscoring.

`tracing.py` must import neither `groq` nor `openai`. `_provider_error_trace_fields` currently reads an exception's type name and HTTP status generically — confirm with `sed -n '126,136p'` that it does not reference an SDK class; if it does, move that one branch into `transport.py` and have `tracing` take the already-extracted fields.

- [ ] **Step 4: Extract transport, then reduce the orchestrator**

`transport.py` takes the SDK imports and everything that touches them: `GROQ_MODEL`, `OPENAI_MODEL`, `MAX_COMPLETION_TOKENS`, `_RATE_LIMIT_ERROR_TYPES`, `_JSON_SCHEMA_RETRY_ERROR_TYPES`, `_extract_retry_after_seconds`, `_messages`, `_is_unsupported_response_format_error`, and — as methods of a new `ProviderTransport` — `_key_fingerprint`, `_sdk_client`, `aclose`, `_invoke`, `_call_groq`, `_call_openai`. `ProviderTransport.__init__(self, *, trace_hook: TraceHook | None = None)` owns `_clients` and `_lock`; `_emit` moves with it.

`RATE_LIMIT_BACKOFF_SECONDS` stays in `groq_openai_client.py` — it is failover policy, not transport.

`GroqOpenAiLlmClient` then holds `self._transport = ProviderTransport(trace_hook=trace_hook)`, and:

```python
    async def aclose(self) -> None:
        """Lifespan/runner teardown, delegated. Idempotent: closing twice, or
        closing a client that never made a call, closes nothing and raises
        nothing. Every cached client is attempted even if an earlier close
        fails; failures aggregate into an ExceptionGroup."""
        await self._transport.aclose()
```

Finish with the compatibility re-exports at the bottom of `groq_openai_client.py`:

```python
from src.adapters.secondary.llm.tracing import LlmTraceEvent, TraceHook, log_llm_trace  # noqa: F401
from src.adapters.secondary.llm.transport import GROQ_MODEL, OPENAI_MODEL  # noqa: F401

__all__ = [
    "GROQ_MODEL",
    "OPENAI_MODEL",
    "RATE_LIMIT_BACKOFF_SECONDS",
    "GroqOpenAiLlmClient",
    "LlmTraceEvent",
    "TraceHook",
    "log_llm_trace",
]
```

Put these imports at the top of the file in the normal position — the snippet shows them together only to make the required set explicit. `ruff` will enforce placement.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_llm_module_boundaries.py -q`

Expected: PASS. `tests/test_llm_adapter.py` will still be RED at this point because its SDK patch targets moved — Task 3 fixes that, and this step deliberately does not.

---

### Task 3: Re-point the SDK patch targets, pin the safety contracts, sync the docs

**Files:**
- Modify: `tests/test_llm_adapter.py` (every `patch(...)` target), `tests/test_llm_client_json_repair.py`, `CLAUDE.md`
- Test: `tests/test_llm_adapter.py`

**Interfaces:**
- Consumes: `transport.groq.AsyncGroq` / `transport.openai.AsyncOpenAI` as the new single patch point.
- Produces: no runtime surface. Pins secret non-disclosure, single-owner teardown, and safe error surfaces so the split cannot silently regress them.

- [ ] **Step 1: Write the failing contract tests** — append to `tests/test_llm_adapter.py`:

```python
def test_trace_events_never_carry_a_key_or_prompt_text() -> None:
    """LlmTraceEvent reaches production stdout through log_llm_trace, so its
    field list is a disclosure boundary, not an internal record."""
    from dataclasses import fields

    from src.adapters.secondary.llm.tracing import LlmTraceEvent

    names = {f.name for f in fields(LlmTraceEvent)}
    assert not (names & {"system_prompt", "user_prompt", "question", "answer", "api_key", "detail", "message"})


def test_aclose_is_idempotent_and_closes_nothing_when_unused() -> None:
    client = _client()
    _run(client.aclose())
    _run(client.aclose())


def test_provider_error_surface_carries_type_and_status_only() -> None:
    """A provider's error body can echo prompt fragments back, so str(exc) is
    deliberately absent from the trace."""
    from src.adapters.secondary.llm.tracing import provider_error_trace_fields

    exc = RuntimeError("leaked: the secret question text")
    fields_out = provider_error_trace_fields(exc)

    assert "leaked" not in repr(fields_out)
    assert fields_out.get("error_type") == "RuntimeError"
```

Check the real key name `provider_error_trace_fields` returns (`sed -n '126,136p'` on the pre-split file) and match the final assertion to it rather than assuming `error_type`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_llm_adapter.py -q`

Expected: FAIL broadly — every SDK patch target still names `groq_openai_client.groq` / `groq_openai_client.openai`, which no longer exist there.

- [ ] **Step 3: Re-point every patch target**

```bash
grep -rn "groq_openai_client\.groq\|groq_openai_client\.openai" tests/
```

Each becomes `src.adapters.secondary.llm.transport.groq.AsyncGroq` / `...transport.openai.AsyncOpenAI`. Patches of `GroqOpenAiLlmClient` internals (`_generate_structured_impl`, `_get_provider_response`) stay where they are — those methods did not move.

- [ ] **Step 4: Update `CLAUDE.md`'s mocking convention and module notes**

Two edits:

1. The Conventions "Mocking convention" bullet names `groq_openai_client.groq.AsyncGroq` / `groq_openai_client.openai.AsyncOpenAI` as the patch targets. Replace with `transport.groq.AsyncGroq` / `transport.openai.AsyncOpenAI`, and add one sentence saying the SDK boundary is now a single module (`src/adapters/secondary/llm/transport.py`) precisely so there is one place to patch.
2. The Conventions "Modular monolith with ports/adapters" bullet describes the five `Protocol`s. Note that `LLMClientPort.generate_structured` takes `(system_prompt, user_prompt, schema)` and that provider + `SecretStr` credentials are constructor-injected, so `src/domain/` no longer imports `src/core/config`.

- [ ] **Step 5: Run the full bucket verification**

```bash
pytest -q
ruff check src tests
mypy src
```

Expected: all green. Then confirm the boundary held:

```bash
grep -rn "^import groq\|^import openai" src/adapters/secondary/llm/
```

Expected: `transport.py` only. Stop at green — committing needs the owner's separate authorization.
