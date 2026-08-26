---
title: Manufacturing Knowledge RAG Assistant
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
---

# Manufacturing Knowledge RAG Assistant

A bilingual (English/Spanish), citation-mandatory Retrieval-Augmented Generation assistant over manufacturing SOPs, equipment manuals, and quality/safety procedures — built to demonstrate applied AI on industrial data with real, inspectable evidence, not a generic "chat with your PDF" demo.

**This is a static showcase page, not the live app.** The live demo link below will be filled in once deployment completes (see status note on the page).

## What makes this different

- **Citations are never trusted from the LLM.** Every citation is re-derived from the real retrieved chunk's metadata by `chunk_id` — the LLM only ever supplies the ID, never the document title, section, or revision. This is the project's core anti-hallucination guarantee.
- **Threshold-based refusal, not vibes.** A calibrated cosine-similarity gate decides whether there's enough retrieval confidence to answer at all, before generation even runs.
- **Hybrid retrieval**: BM25 lexical search fused with `BAAI/bge-m3` multilingual dense embeddings via Reciprocal Rank Fusion.
- **Honest data**: the corpus mixes real public-domain U.S. federal documents (OSHA, DOE, NIOSH, CFR) with clearly-labeled synthetic documents — every source is labeled and verified, never presented as something it isn't.
- **Bilingual by design**: ask in English or Spanish; answers come back in the question's language; citations always point to the original English source.

See the full architecture, eval results, and design rationale on the page. Source code: [GitHub — manufacturing-rag-assistant](https://github.com/Jehu-Lara/manufacturing-rag-assistant).
