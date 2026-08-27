# ADR-003: Ports and interfaces at the domain boundary

## Context

Before this refactor, code called concrete modules directly —
`retrieval.vector_store.query`, `api.llm_client.generate_structured` — so
swapping an implementation or writing a real (non-framework-mocked) unit test
meant monkeypatching module attributes. `api/generation.py`'s tests, for
example, patched `retrieval.hybrid.retrieve` and `api.llm_client.generate_structured`
as module-level functions rather than exercising a real interface.

## Decision

Define five minimal `Protocol`s in `src/domain/ports.py`:
`RetrieverPort`, `LLMClientPort`, `VectorStorePort`, `EmbedderPort`,
`LexicalIndexPort`. Each exposes only the methods actually called elsewhere
in the codebase today — no speculative surface. Structural typing
(`Protocol`, not `ABC`) means adapters implement these by shape, not by
inheritance; the composition root (`src/main.py` / the HTTP adapter's
`lifespan`) is the only place concrete adapters are constructed and wired to
a port-typed parameter.

`VectorStorePort` gains one method — `ping() -> bool` — that isn't a literal
1:1 mapping of an existing call: today's `/health` handler calls
`retrieval.vector_store.get_collection()` directly and catches its exception
inline. Exposing ChromaDB's raw `Collection` type through the port would leak
an adapter type across the domain boundary — exactly what the import-invariant
test exists to catch — so the port abstracts it as a boolean instead.

## Consequences

- Tests get real in-memory port fakes (`InMemoryRetriever`, `InMemoryLLMClient`)
  instead of patching module-level functions — behavior tests without
  framework mocks, per this project's testing philosophy.
- The import-invariant test (`tests/test_import_invariants.py`) is only
  meaningful because `domain/ports.py` never imports a concrete adapter
  library — the ports are the seam that makes "domain imports nothing from
  fastapi/chromadb/groq/openai/streamlit/torch" an enforceable rule rather
  than a convention.
- Adding a new adapter (e.g. swapping the LLM provider or the vector store)
  means implementing a port, not touching `domain/` or the use cases that
  depend on it.
