# Architecture — as-built (post-refactor) vs. target

This repo has already reached the target modular-monolith shape for
everything except the deploy container split (deferred — see ADR-005 and
the migration plan's Phase 4b appendix). The diagrams below reflect that:
"as-built" and "target" now differ only in the deploy layer.

## Container diagram — as-built (current, real)

```mermaid
graph TB
    operator["Operator / demo visitor<br/>(browser)"]

    subgraph container["Single Docker container (Hugging Face Space, port 7860)"]
        nginx["nginx<br/>reverse proxy"]
        streamlit["Streamlit process<br/>src/web/app.py"]
        api["FastAPI process<br/>src.main:app (uvicorn)"]
        chroma[("ChromaDB<br/>persistent client<br/>baked into image")]
        bm25[("BM25 JSON index<br/>baked into image")]
    end

    llm["Groq / OpenAI<br/>LLM API"]

    operator -->|HTTPS| nginx
    nginx -->|"/ (websocket)"| streamlit
    nginx -->|"/query, /health, /ready"| api
    streamlit -->|"httpx, HTTP-only"| nginx
    api --> chroma
    api --> bm25
    api -->|"structured generation"| llm
```

## Component diagram — `src/` modular monolith (as-built)

```mermaid
graph LR
    subgraph web["src/web/ — Streamlit, HTTP-only"]
        webapp["app.py"]
        client["client.py"]
        i18n["i18n.py"]
        render["render.py"]
    end

    subgraph primary["src/adapters/primary/http/"]
        router["src/features/query/router.py"]
        appfactory["app.py — create_app(), lifespan"]
    end

    subgraph domain["src/domain/ — zero framework imports"]
        models["models.py"]
        policies["policies.py — RRF, RefusalPolicy, CitationResolver"]
        ports["ports.py — 5 Protocols"]
    end

    subgraph features["src/features/"]
        query_uc["query/use_cases.py — QueryUseCase"]
        retrieval_uc["retrieval/use_cases.py — HybridRetriever"]
        ingestion_f["ingestion/"]
        evaluation_f["evaluation/"]
    end

    subgraph secondary["src/adapters/secondary/"]
        embedder["embedder/ — SentenceTransformersEmbedder"]
        vector["vector/ — ChromaVectorStore"]
        lexical["lexical/ — Bm25LexicalIndex (JSON)"]
        llmadapter["llm/ — GroqOpenAiLlmClient (async)"]
    end

    webapp --> client
    webapp --> i18n
    webapp --> render
    client -->|"HTTP only"| router

    router --> query_uc
    query_uc --> ports
    query_uc --> retrieval_uc
    retrieval_uc --> ports
    query_uc -.implements.-> llmadapter
    retrieval_uc -.implements.-> vector
    retrieval_uc -.implements.-> lexical
    vector --> embedder

    appfactory --> router
    ingestion_f --> models
    evaluation_f --> retrieval_uc
    evaluation_f --> query_uc
```

## Deploy layer — target (Phase 4b, deferred)

```mermaid
graph TB
    operator["Operator / demo visitor"]
    subgraph api_container["api container"]
        apionly["uvicorn src.main:app<br/>ONLY"]
    end
    subgraph web_container["web container"]
        webonly["Streamlit<br/>src/web/app.py"]
    end
    volume[("Mounted volume<br/>Chroma + BM25 JSON<br/>NOT baked into image")]

    operator --> webonly
    operator --> apionly
    webonly -->|httpx| apionly
    apionly --> volume
```

Not implemented: blocked on choosing a deploy target that supports
multi-service compose and persistent volumes (today's HF Docker Space free
tier supports neither), per ADR-005.
