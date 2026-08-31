# Results — Bilingual / terse-query false-refusal fix (Phase 1 measurement + decision gate)

- **Date basis:** git commit `a0c0a8cfa8d35cb3202cea30de375df488114f71`, committed `2026-08-30T12:42:51-06:00`
- **eval_set:** v1.1.0, stored SHA-256 `d846b3dc25f4bae6ba749ff5005d72e8199735c0b637c5b4fe8eb65ee54bee8c` (105 questions: 48 en / 32 es answerable, 10 en / 15 es unanswerable)
- **regression_set:** v1.0.0, stored SHA-256 `51874427db66723820fe7164845cae8dd0e7f9e59babf5f357312671bf4234e5` (20 queries)
- **`REFUSAL_COSINE_THRESHOLD`:** `0.5999` — unchanged. This document measures; it does not select or ship a threshold.
- **Immutability:** `eval/reports/*_v1.0.0.md` were not regenerated or edited. All new output is `*_v1.1.0*.md`.
- **Companion design:** `docs/superpowers/specs/2026-08-29-bilingual-refusal-fix-design.md` §6 (frozen acceptance criteria).

## 0. Index-staleness check (precondition)

`retrieval_report_v1.1.0.md` (baseline, `expansion_mode=off`) was compared against `retrieval_report_v1.0.0.md`
on the 30 questions shared by both eval-set versions (`q001`–`q030`):

- max |top-1 semantic-score delta| across all 30 = **0.00000** (byte-identical)
- Recall@5 on those 30 = **0.833** in both versions

The built index at `retrieval/output/` is **not stale**. No reindex was performed, and none was needed
(this plan changed no corpus file).

## 1. What was run

| Command | Output | Cost |
|---|---|---|
| `python -m src.features.evaluation.retrieval_eval` (baseline `off`) | `retrieval_report_v1.1.0.md`, copied to `…__off.md` | embeddings only |
| `python -m src.features.evaluation.threshold_analysis` | `threshold_analysis_v1.1.0.md` (pooled + per-language sweeps) | embeddings only |
| `python -m src.features.evaluation.regression_eval` | `regression_eval_v1.1.0.md` (all 4 modes, one section each) | embeddings only |
| `run('semantic')`, `run('lexical')`, `run('both')` on `retrieval_eval` | `retrieval_report_v1.1.0__<mode>.md` | embeddings only |
| `generation_eval` | **not run — see §2** | would be paid LLM calls |

`retrieval_report_v1.1.0.md` (the canonical, unsuffixed file) holds the **baseline `off`** run;
the three interventions live in the `__<mode>` siblings so all four configs' numbers survive.

## 2. `generation_eval` was NOT run — all refusal numbers below are gate-only

*Original Task 8 state:* `GROQ_API_KEY` was rejected with `401 … {"code": "invalid_api_key"}` and
`OPENAI_API_KEY` was unset, so the baseline `generation_eval` aborted with every question returning
*"structured generation failed on all providers"*.

*Phase 1 closeout update:* the owner rotated the Groq key; it now authenticates. A fresh
`generation_eval` run (`off` then `semantic`) was started and, after ~90 minutes of repeated
**`429` rate-limiting** on the Groq free tier (`"rate limited, backing off before retrying same
provider"` on loop), was aborted with no `generation_eval_v1.1.0.md` produced. `OPENAI_API_KEY` is
still unset, so there is no fallback. Filling the four ⚠️ cells needs the Groq limits to reset, a paid
tier, or the OpenAI fallback key — see §6 follow-up 1. **The key was never edited or printed by any task.**

Consequently, every correct-refusal and false-refusal figure in this document is a
**gate-only estimate**: the refusal decision predicted by applying `0.5999` to each question's
top-1 pure-semantic cosine score, with **LLM self-refusal not measured**.

**Gate-only estimates are known to be biased, and the bias is not small.** On v1.0.0 the gate alone
predicts 2/30 false refusals (0.067) and 7/10 correct refusals (0.700); the real
`generation_eval_v1.0.0.md` measured **0.200** false-refusal and **0.900** correct-refusal. The LLM's own
"insufficient information" behaviour accounted for 4 extra false refusals and 2 extra correct refusals.
So gate-only figures **understate false-refusal** and **understate correct-refusal**. Any acceptance cell
resting on them is *unverified*, not *passed*.

## 3. Matched-pair EN−ES cosine gap — answers the design's open question

Design §2.4 flagged "ES runs ~0.05–0.09 below EN" as an **open question**, to be settled by the frozen
matched-pair data before any per-language threshold is considered. Settled:

| config | mean gap (en − es), n=25 pairs |
|---|---|
| `off` (baseline) | **+0.0426** |
| `semantic` | **+0.0298** |
| `lexical` | +0.0426 (identical to baseline — BM25 expansion cannot move a cosine score) |
| `both` | +0.0298 |

**Spanish does run below English, but by less than the hypothesis claimed and within the design's
`<= 0.05` target at every config.** 22 of 25 pairs have a positive gap; 3 are negative (`q078`/`q077`
−0.0232, `q088`/`q087` −0.0113, `q090`/`q089` −0.0152), so the effect is a consistent shift, not a
uniform one. The largest single gaps are `q062`/`q061` +0.1441, `q052`/`q051` +0.1136, `q060`/`q059` +0.1020.

This gap is **evidence, not a gate** (design §6). It does not on its own justify a per-language threshold:
a +0.043 mean shift is smaller than the 0.09-wide band (0.5284–0.6194) in which the bulk of the
gate-over-refusals sit, so the dominant problem is where the single threshold is drawn relative to
*both* languages' answerable distributions, not the EN/ES delta.

## 4. Acceptance table (design §6), per config

Hard cells are marked **H**. `off` is the current production behaviour.

| config | EN Recall@5 (**H** >= 0.913) | ES Recall@5 (**H** >= 0.80) | matched-pair gap (evidence, target <= 0.05) | EN correct-refusal (**H** >= 0.90) | ES correct-refusal (**H** >= 0.80) | EN false-refusal (**H** <= 0.10) | ES false-refusal (soft) | r001/r002 answered (**H**) |
|---|---|---|---|---|---|---|---|---|
| `off` | 0.917 ✅ | **0.781 ❌** | +0.0426 ✅ | 0.700 ⚠️ gate-only | 0.933 ⚠️ gate-only | **0.125 ❌** gate-only | 0.375 gate-only | **no / no ❌** |
| `semantic` | 0.917 ✅ | **0.781 ❌** | +0.0298 ✅ | 0.700 ⚠️ gate-only | 0.933 ⚠️ gate-only | 0.062 ⚠️ gate-only | 0.219 gate-only | **yes / yes ✅** |
| `lexical` | 0.938 ✅ | **0.688 ❌** | +0.0426 ✅ | 0.700 ⚠️ gate-only | 0.933 ⚠️ gate-only | **0.125 ❌** gate-only | 0.375 gate-only | **no / no ❌** |
| `both` | 0.917 ✅ | **0.688 ❌** | +0.0298 ✅ | 0.700 ⚠️ gate-only | 0.933 ⚠️ gate-only | 0.062 ⚠️ gate-only | 0.219 gate-only | **yes / yes ✅** |

⚠️ = the cell's own metric was not measurable this run (§2); the figure shown is the gate-only proxy and
is **not** a pass.

Supporting retrieval detail (all from `retrieval_report_v1.1.0__<mode>.md`):

| config | EN R@3 | EN R@5 | ES R@3 | ES R@5 | pooled R@3 | pooled R@5 | pooled MRR |
|---|---|---|---|---|---|---|---|
| `off` | 0.792 | 0.917 | 0.625 | 0.781 | 0.725 | 0.863 | 0.660 |
| `semantic` | 0.812 | 0.917 | 0.656 | 0.781 | 0.750 | 0.863 | 0.669 |
| `lexical` | 0.833 | 0.938 | 0.656 | 0.688 | 0.762 | 0.838 | 0.648 |
| `both` | 0.833 | 0.917 | 0.656 | 0.688 | 0.762 | 0.825 | 0.656 |

### 4a. The reported queries (r001 / r002), per config

From `regression_eval_v1.1.0.md`. Both target `doe-hdbk-1018-1-pumps::chunk-0007`, which is fused-rank 1
in **every** config — retrieval was never the failure here.

| config | r001 (en) top-1 cosine → gate | r002 (es) top-1 cosine → gate |
|---|---|---|
| `off` | 0.5582 → **REFUSE** | 0.5598 → **REFUSE** |
| `semantic` | 0.7195 → **answer** | 0.7193 → **answer** |
| `lexical` | 0.5582 → **REFUSE** | 0.5598 → **REFUSE** |
| `both` | 0.7195 → **answer** | 0.7193 → **answer** |

Query expansion fixes the two user-reported queries, and only through the **semantic** channel.
`lexical` leaves the gate scalar untouched by construction (the gate reads the cosine score, which
BM25 never contributes to), so `lexical` cannot ever fix a gate over-refusal — it can only move rankings,
and here it moves them the wrong way (§4b).

### 4b. Cost of expansion on the frozen regression controls

| config | answerable passing gate | unanswerable controls correctly refused |
|---|---|---|
| `off` | 9/17 | **3/3** |
| `semantic` | 14/17 | **1/3** |
| `lexical` | 9/17 | **3/3** |
| `both` | 14/17 | **1/3** |

`r018` (`What NPSH margin does API 610 recommend for this pump?`) and `r019` (its Spanish twin) are the
control unanswerables the design deliberately built to *look* on-topic — API 610 is not in the corpus.
Expansion pushes them from 0.5623/0.5018 to 0.6719/0.6638, flipping both from **REFUSE to answer**.
This is the mechanism working exactly as designed and producing exactly the side effect the design
worried about: appending corpus-attested glossary text to a query raises its similarity to corpus chunks
*whether or not the corpus actually answers it*. The eval-set unanswerable subset does not show this
(gate-only correct-refusal is flat at EN 0.700 / ES 0.933 across all four configs) **only because none of
its 25 unanswerables contain a glossary acronym** — the regression set is the sharper instrument here,
and it says expansion costs real refusal precision.

## 5. Classification table

Every v1.1.0 answerable miss and every regression failure at `expansion_mode=off`.
Classes: `gate-over-refusal` (expected chunk in top-5, gate refuses), `retrieval-miss` (expected
document absent from top-5), `decoy-chunk` (a sibling/related chunk outranks the expected one, expected
document still present in top-5).

The decoy-chunk vs retrieval-miss split is reproduced in the committed artifact
`eval/reports/classification_v1.1.0__raw-v1__off.md` (a re-run of retrieval over eval_set v1.1.0
with a deterministic classifier). It confirms **11 non-gate failures = 9 same-document decoys +
2 cross-document decoys, 0 true retrieval misses**, and separately **15 gate-over-refusals**
(retrieval OK, gate refuses — a larger group than the decoys). Phase 2 Task 3 replaces that
artifact with a per-question JSONL top-5 dump.

### 5a. eval_set v1.1.0 answerable misses (recall@5 == 0), `off` — n=11

| id | language | retrieval OK? | gate decision | class |
|---|---|---|---|---|
| q002 | en | no (`osha-3120-lockout-tagout::chunk-0002` top-1, expected `::chunk-0018`) | answer (0.7397) | decoy-chunk |
| q014 | es | no (`cfr-21-part-211-cgmp::chunk-0001` top-1, expected `::chunk-0024`) | answer (0.6124) | decoy-chunk |
| q017 | en | no (`niosh-pocket-guide-excerpt::chunk-0006` top-1, expected `::chunk-0010`) | answer (0.6618) | decoy-chunk |
| q018 | es | no (`niosh-pocket-guide-excerpt::chunk-0007` top-1, expected `::chunk-0018`) | answer (0.6790) | decoy-chunk |
| q026 | es | no (`osha-3170-machine-guarding::chunk-0003` top-1, expected `::chunk-0021`) | answer (0.6855) | decoy-chunk |
| q050 | en | no (`osha-3170-machine-guarding::chunk-0021` top-1, expected `osha-3120-lockout-tagout::chunk-0016`) | REFUSE (0.5674) | decoy-chunk (cross-document) |
| q051 | es | no (`osha-3170-machine-guarding::chunk-0004` top-1, expected `osha-3120-lockout-tagout::chunk-0012`) | answer (0.6130) | decoy-chunk (cross-document) |
| q066 | en | no (`niosh-pocket-guide-excerpt::chunk-0006` top-1, expected `::chunk-0015`) | answer (0.6227) | decoy-chunk |
| q067 | es | no (`niosh-pocket-guide-excerpt::chunk-0007` top-1, expected `::chunk-0012`) | answer (0.6360) | decoy-chunk |
| q075 | es | no (`cfr-21-part-211-cgmp::chunk-0001` top-1, expected `::chunk-0014`) | REFUSE (0.5718) | decoy-chunk |
| q083 | es | no (`sop-mnt-022-cnc-mill-changeover::chunk-0002` top-1, expected `::chunk-0005`) | REFUSE (0.5964) | decoy-chunk |

**All 11 are decoy-chunk. Zero are pure retrieval misses** — in every case the expected *document*
appears in the top-5 and a sibling section outranks the expected chunk. 9 of 11 have the decoy in the
same document as the expected chunk.

### 5b. eval_set v1.1.0 gate-over-refusals (recall@5 == 1 but gate refuses), `off` — n=15

Retrieval is correct; the gate rejects it. 9 es / 6 en.

| id | language | top-1 semantic | id | language | top-1 semantic |
|---|---|---|---|---|---|
| q009 | es | 0.5796 | q071 | es | 0.5325 |
| q024 | en | 0.5837 | q072 | en | 0.5419 |
| q041 | es | 0.5284 | q073 | es | 0.5979 |
| q042 | en | 0.5377 | q085 | es | 0.5286 |
| q049 | es | 0.5470 | q089 | es | 0.5677 |
| q059 | es | 0.5929 | q090 | en | 0.5525 |
| q065 | es | 0.5788 | | | |
| q069 | es | 0.5635 | | | |
| q070 | en | 0.5798 | | | |

Under `semantic`/`both` this set shrinks to 8 (`q041`, `q042`, `q049`, `q065`, `q069`, `q070`, `q073` recover;
none is newly added).

### 5c. regression_set failures at `off` — n=9

| id | language | retrieval OK? | gate decision | class |
|---|---|---|---|---|
| r001 | en | yes (recall@5 = 1.0) | REFUSE (0.5582) | gate-over-refusal |
| r002 | es | yes | REFUSE (0.5598) | gate-over-refusal |
| r006 | en | yes | REFUSE (0.4988) | gate-over-refusal |
| r007 | es | yes | REFUSE (0.5011) | gate-over-refusal |
| r009 | es | yes | REFUSE (0.5796) | gate-over-refusal |
| r010 | en | no (top-5 all `niosh-pocket-guide-excerpt`, expected `::chunk-0010` absent) | answer (0.6131) | decoy-chunk |
| r011 | es | yes | REFUSE (0.5303) | gate-over-refusal |
| r014 | en | yes | REFUSE (0.5876) | gate-over-refusal |
| r017 | en | yes | REFUSE (0.5285) | gate-over-refusal |

8 of 9 are gate-over-refusals; the one retrieval failure is again a decoy chunk inside the right document.

### 5d. What the classification says

Two distinct, non-overlapping failure modes, and **query expansion only addresses one of them**:

1. **Gate over-refusal** (15 eval + 8 regression at `off`): the correct chunk is retrieved and the
   `0.5999` cutoff rejects it. Query expansion raises the cosine score and recovers about half of these
   (15 → 8 on the eval set, 8 → 3 on the regression set — only `r009`, `r011`, `r014` still refuse) —
   at the cost of also lifting genuine
   unanswerables over the line (§4b).
2. **Decoy chunk** (11 eval + 1 regression at `off`): the right *document* is retrieved and a sibling
   *section* outranks the right one. Expansion does not fix this — the miss count is 11 at `off`, 11 at
   `semantic`, 13 at `lexical`, 14 at `both`. This is a **chunk-discrimination** problem: the embedding
   of a bare chunk body carries no signal about which section of which document it is. That is precisely
   the failure Phase 2 (C2, heading-prefixed embedding input) is designed for.

## 6. DECISION

> **No C1 configuration clears the frozen acceptance table. Proceed to the Phase 2 (contextual chunk
> embedding) plan.**

### Cells that fail, and by how much

| failing hard cell | best config | value | required | shortfall |
|---|---|---|---|---|
| **ES Recall@5 >= 0.80** | `off` / `semantic` (tie) | **0.781** (25/32) | 0.80 | **1 question** — 26/32 = 0.8125 would clear it. `lexical`/`both` are far worse at 0.688 (22/32). |
| **EN correct-refusal >= 0.90** | — | **not measurable** | 0.90 | No LLM credential (§2). Gate-only proxy is 0.700; on v1.0.0 the same proxy read 0.700 where the measured value was 0.900, so this cell is genuinely unknown, not failed. |
| **ES correct-refusal >= 0.80** | — | **not measurable** | 0.80 | Same. Gate-only proxy 0.933. |
| **EN false-refusal <= 0.10** | `semantic` / `both` | **not measurable** | 0.10 | Gate-only proxy 0.062 (down from 0.125 at `off`), but the v1.0.0 calibration shows the real figure runs ~3× the proxy. Cannot be claimed as passed. |
| **r001/r002 answered** | `semantic` / `both` | **passes** | — | The only hard cell any intervention decisively clears. |

`semantic` is unambiguously the best of the three interventions — it is the only one that fixes the two
reported queries without degrading any retrieval metric (EN R@5 0.917 unchanged, ES R@5 0.781 unchanged,
pooled R@3 0.725 → 0.750, MRR 0.660 → 0.669, gap +0.0426 → +0.0298). `lexical` and `both` are strictly
worse: both drop ES Recall@5 from 0.781 to 0.688 (−3 questions), and `both` drops pooled Recall@5 to 0.825.
**`lexical` and `both` are ruled out on retrieval evidence alone.**

But `semantic` still does not ship, for three independent reasons:

1. **ES Recall@5 fails at 0.781** and expansion does not move it — the ES ceiling is set by decoy chunks
   (§5d.2), which C1 cannot touch.
2. **It costs unanswerable precision.** The frozen controls go 3/3 → 1/3 correctly refused (§4b). Shipping
   an intervention that turns two deliberately-planted "looks on-topic, isn't in the corpus" queries into
   answered ones is directly contrary to this project's no-silent-hallucination standard, and the eval
   set's unanswerable subset is blind to it.
3. **Half the acceptance table was not measurable this run.** Shipping against an unverified
   correct-refusal cell is not a decision, it is a guess.

### Follow-ups (each its own reviewed increment — none performed here)

1. **Re-measure `generation_eval` once the LLM provider is usable.** *Update (Phase 1 closeout):* the Groq
   key was rotated by the owner and now authenticates, but `generation_eval` over 105 questions is
   **429 rate-limited** on the Groq free tier — a run backed off repeatedly for ~90 min and was aborted
   with no report produced. The four ⚠️ refusal cells stay gate-only until the limits reset, a paid tier is
   configured, or the OpenAI fallback key is set. Phase 2 Task 8 retries `generation_eval` behind its own
   owner gate; the Phase 1 no-C1-config-ships decision does not depend on it (it rests on ES Recall@5 = 0.781,
   a retrieval number).
2. **Phase 2 (C2, contextual chunk embedding)** — design §6 steps 11–13, seeded by §5 above: **11 non-gate
   failures = 9 same-document decoys + 2 cross-document decoys, 0 true retrieval misses**
   (`eval/reports/classification_v1.1.0__raw-v1__off.md`). The 9 same-document decoys are the exact target of
   prefixing `document_title > section_heading` onto the embedding input while leaving `documents=` as raw
   `chunk_text`; the 2 cross-document decoys are a weaker fit. Requires ADR-008 and a full
   `python -m src.features.retrieval.cli` reindex. Separately, **15 gate-over-refusals** (retrieval OK, gate
   refuses — larger than the decoy set) will need a gate follow-up (design §6 Phase 3), not C2.
3. **Re-measure C1 on top of C2** once C2 lands — the two interventions attack different failure modes and
   the gate-over-refusal set may look different after reindexing.
4. **Do not change `REFUSAL_COSINE_THRESHOLD`.** It stays `0.5999`. `threshold_analysis_v1.1.0.md` reports a
   pooled selection of 0.5271 on v1.1.0 (vs 0.5999 on v1.0.0), but that is the analyzer's own rule applied
   to a differently-shaped set, not a recommendation, and Phase 3 is explicitly gated behind Phases 1–2.
5. **Do not flip the production `expansion_mode` default.** It stays `off`.

### Not done, deliberately

No push, no PR, no deploy, no threshold change, no production-default change, no `.env` edit,
no corpus change, no `*_v1.0.0.md` modification.

---

# Phase 2 — contextual chunk embedding (`contextual-v1`): measurement + decision

- **Date basis:** commit `8d7cecd` (production-default flip) on `fix/bilingual-refusal-phase2`; measurements from `eval/reports/*_v1.1.0__contextual-v1__*` (index build commit `ee928ae`).
- **Index built:** `contextual-v1` — each chunk embedded as `f"{document_title} > {section_heading}\n\n{chunk_text}"`; `documents=` and `metadata["chunk_text"]` stored **byte-identical to raw** (`tests/test_chroma_vector_store.py`). BM25 unchanged. Manifest: `index_profile=contextual-v1`, `chunks_sha256=dd7d21b5…`, `corpus_sha256=2fc786ca…`, `embedding_revision=5617a9f6…`.
- **`REFUSAL_COSINE_THRESHOLD` `0.5999`, RRF `k=60`, `expansion_mode` default `off` — all unchanged.**
- **`generation_eval`: NOT run.** The owner elected to close Phase 2 gate-only (Groq free tier 429s a 105-question run; §2). **Every correct-/false-refusal figure below is the gate-only proxy** (refuse ⇔ top-1 pure-semantic cosine < 0.5999) and is **UNVERIFIED**, not passed — the §2 calibration (proxy understates both false- and correct-refusal) still applies. This section does not claim generation quality or citation integrity were re-validated on `contextual-v1`.
- **ADR:** `docs/adr/008-contextual-chunk-embedding.md` (Accepted 2026-08-30).

## 7. Four-config comparison

Retrieval (eval_set v1.1.0 — 80 answerable [EN 48 / ES 32], 25 unanswerable). **H** = hard acceptance cell.

| config | EN R@3 | EN R@5 (**H** ≥ 0.913) | ES R@3 | ES R@5 (**H** ≥ 0.80) | pooled R@3 | pooled R@5 | pooled MRR |
|---|---|---|---|---|---|---|---|
| `raw-v1` / `off` (previous production) | 0.792 | 0.917 ✅ | 0.625 | **0.781 ❌** | 0.725 | 0.863 | 0.660 |
| `raw-v1` / `semantic` (Phase 1) | 0.812 | 0.917 ✅ | 0.656 | **0.781 ❌** | 0.750 | 0.863 | 0.669 |
| **`contextual-v1` / `off`** (new production) | 0.896 | **0.917 ✅** | 0.719 | **0.844 ✅** | 0.825 | 0.887 | 0.721 |
| `contextual-v1` / `semantic` | 0.917 | 0.917 ✅ | 0.688 | 0.812 ✅ | 0.825 | 0.875 | 0.711 |

Refusal — **gate-only proxy at 0.5999, UNVERIFIED**:

| config | answerable wrongly refused (pooled) | — EN (**H** ≤ 0.10) | — ES (soft) | unanswerable correctly refused (pooled) | matched-pair gap (evidence, ≤ 0.05) |
|---|---|---|---|---|---|
| `raw-v1` / `off` | 18/80 = 0.225 | 6/48 = 0.125 ⚠️ | 12/32 = 0.375 ⚠️ | 21/25 = 0.84 ⚠️ | +0.0426 ✅ |
| **`contextual-v1` / `off`** | 9/80 = 0.113 | 3/48 = 0.063 ⚠️ | 6/32 = 0.188 ⚠️ | 21/25 = 0.84 ⚠️ | +0.0449 ✅ |
| `contextual-v1` / `semantic` | 3/80 = 0.038 | 1/48 = 0.021 ⚠️ | 2/32 = 0.063 ⚠️ | 21/25 = 0.84 ⚠️ | +0.0298 ✅ |

⚠️ = gate-only proxy; the metric's real value was not measured and is **not** a pass.

Frozen regression set (20 queries) — gate-only:

| config | answerable passing gate | API-610 controls `r018`/`r019`/`r020` correctly refused | `r001` (en) → gate | `r002` (es) → gate |
|---|---|---|---|---|
| `raw-v1` / `off` (Phase 1 frozen baseline) | 9/17 | **3/3 ✅** | 0.5582 → REFUSE | 0.5598 → REFUSE |
| `raw-v1` / `semantic` (Phase 1) | 14/17 | **1/3 ❌** | 0.7195 → answer | 0.7193 → answer |
| **`contextual-v1` / `off`** | 11/17 | **3/3 ✅** | 0.5642 → REFUSE | 0.5656 → REFUSE |
| `contextual-v1` / `semantic` | 17/17 | **0/3 ❌** | 0.7128 → answer | 0.7124 → answer |

Failure classification (answerable subset, n=80; deterministic classifier over the committed per-question JSONL):

| config | gate-over-refusal | same-document-decoy | cross-document-decoy | retrieval-miss |
|---|---|---|---|---|
| `raw-v1` / `off` | 15 | 9 | 2 | **0** |
| **`contextual-v1` / `off`** | 8 | 6 | 3 | **0** |
| `contextual-v1` / `semantic` | 3 | 7 | 3 | **0** |

(`raw-v1` / `semantic` was not re-run through the JSONL classifier; Phase 1 §5b records it as 8 gate-over-refusals + 11 decoys.)

### 7.1 What `contextual-v1` / `off` does

- **Clears the ES Recall@5 ship gate: 0.781 → 0.844** (`q018`, `q067` recovered from same-document decoys). This was the central Phase 2 objective.
- **Holds every other hard retrieval cell.** EN Recall@5 flat at 0.917 (≥ 0.913). Pooled R@3 0.725 → 0.825, MRR 0.660 → 0.721 — the heading path gives the embedder the section-discrimination signal it lacked (ADR-008 hypothesis confirmed on the frozen evidence).
- **No gate-only refusal-precision regression.** Unanswerable-correctly-refused flat at 21/25 pooled — identical to `raw-v1`. API-610 controls stay **3/3**. The prefix raises the cosine of answerable content without raising the cosine of the planted "looks on-topic, isn't in the corpus" queries.
- **Halves the gate-only false-refusal proxy** (18/80 → 9/80 pooled; EN 6→3, ES 12→6) as a side effect of the cosine lift; recovers 7 of the 15 eval gate-over-refusals and regression `r009`/`r014`.
- **Citation / prompt payload unchanged.** The contextual prefix never reaches Chroma `documents=`, `metadata["chunk_text"]`, the generation prompt, or `CitationResolver` — verified byte-for-byte by test.

### 7.2 What it does NOT do

- **`r001` / `r002` still refuse.** The target chunk `doe-hdbk-1018-1-pumps::chunk-0007` is retrieved (recall@5 = 1.0), but its top-1 cosine only moves 0.558 → 0.564, still below the 0.5999 gate. A **gate-calibration** problem, not a retrieval problem — ADR-008 "Consequences" explicitly routes the gate-over-refusals to Phase 3.
- **8 gate-over-refusals remain** on the eval set (down from 15).
- **3 cross-document decoys** (`q026`, `q050`, `q051`) — one more than `raw-v1` had; for `q026` a sibling section in another OSHA document now outranks the expected chunk. The weaker fit for C2, as ADR-008 predicted.
- **The generation axis is unverified.** No `generation_eval` on `contextual-v1`.

### 7.3 `contextual-v1` / `semantic` is rejected

ES Recall@5 0.812 and gate-over-refusals down to 3, but the API-610 regression controls go **0/3** correctly refused (worse than `raw-v1` / `semantic`'s 1/3) — query expansion lifts the planted on-topic-looking unanswerables over the gate. Disqualifying against the no-silent-hallucination standard. `expansion_mode` stays `"off"` in production.

### 7.4 Artifact naming

Two naming schemes coexist in `eval/reports/`. The Phase-1 mode-only names
(`retrieval_report_v1.1.0__{off,semantic,lexical,both}.md`, `regression_eval_v1.1.0.md`,
`threshold_analysis_v1.1.0.md`, and the unsuffixed `retrieval_report_v1.1.0.md`
canonical alias) are **frozen Phase-1 artifacts** — do not regenerate them. All
Phase-2-onward output uses the `*__<index_profile>__<expansion_mode>.*` scheme
minted by `src/features/evaluation/artifacts.py` (`artifact_filename`), e.g.
`retrieval_report_v1.1.0__contextual-v1__off.md`,
`classification_v1.1.0__contextual-v1__off.md`. The dataset version (`v1.1.0`)
never moves; profile and mode are suffix axes, not version bumps.

## 8. DECISION (Phase 2)

> **SHIP `contextual-v1` as the new production index profile, `expansion_mode` staying `"off"`.** It clears the ES Recall@5 ship gate (0.781 → 0.844) with no regression to English retrieval, gate-only refusal precision, the API-610 controls, or the citation/prompt payload. The remaining gate-over-refusals — including the user-reported `r001`/`r002` — are a separate gate-calibration problem and move to a dedicated Phase 3 plan.

Decision-branch mapping (plan Task 9):
- **SHIP candidate** — `contextual-v1` / `off`. All hard **retrieval** cells pass. The hard **refusal** cells (EN correct-refusal ≥ 0.90, ES ≥ 0.80, EN false-refusal ≤ 0.10) were not measurable this run; they stay gate-only proxies exactly as at Phase 1 close, and the ship decision rests on the retrieval axis (ES Recall@5), per the plan's Task 0 note.
- **Gate follow-up** — a separate **Phase 3** plan for the 8 gate-over-refusals: global-threshold re-analysis, a two-tier low-confidence response, and/or an evidence-supported per-language threshold. `r001`/`r002` are in scope there. **Do not** change `REFUSAL_COSINE_THRESHOLD` outside that plan.
- **Not Ranking follow-up** — ES Recall@5 now ≥ 0.80 with 0 true retrieval misses, so multilingual sparse retrieval (Phase 2b) is not required to ship.
- **Not Reject/rollback** — no English regression, no citation/payload change, no manifest mismatch, and the API-610 controls did not regress.

### Fixed review format

```text
VEREDICTO: PASS_WITH_FIXES
SEVERIDAD: P2
BLOCKERS:
  - none for shipping contextual-v1 on the retrieval axis.
FIXES_REQUIRED:
  - Refusal % cells stay labelled gate-only / UNVERIFIED. Do not describe the
    contextual-v1 result as "generation-validated" or "production-validated" —
    generation_eval was not run.
  - Open the Phase 3 gate-calibration plan as its own increment before treating
    r001/r002 as resolved.
RESIDUAL_RISKS:
  - r001/r002 and 6 other gate-over-refusals still refuse answerable questions
    whose chunk was retrieved; the 0.5999 gate does not cleanly separate the
    answerable/unanswerable cosine distributions (they overlap ~0.53-0.66).
  - contextual-v1 generation answer quality and citation correctness are
    unmeasured; the raw-chunk-payload invariant is test-verified but end-to-end
    generation was not re-run.
  - 3 cross-document decoys remain (one more than raw-v1) — the weaker fit for C2.
```

### Follow-up increments (none performed in this decision)

1. **Phase 3 — gate calibration.** Its own plan + ADR + owner gate. Owns `r001`/`r002` and the 8 gate-over-refusals. `REFUSAL_COSINE_THRESHOLD` may only change there.
2. **`generation_eval` on `contextual-v1` / `off`** once a non-rate-limited LLM path exists (paid Groq tier or the OpenAI fallback key). Fills the four ⚠️ refusal cells and re-checks citation correctness end-to-end.
3. **`expansion_mode` on top of `contextual-v1`** — done for `semantic` (rejected, §7.3); `lexical`/`both` not re-run and not planned (Phase 1 ruled them out on retrieval evidence).

### Shipped in the Task 9 increment

- `src/features/retrieval/cli.py`: `INDEX_PROFILE` default `raw-v1` → `contextual-v1` (commit `8d7cecd`). Dockerfile + CI build the new profile by default; `INDEX_PROFILE=raw-v1` still rebuilds the rollback index.
- `docs/adr/008-contextual-chunk-embedding.md`: Status `Proposed` → `Accepted`.
- `SPEC.md`, `CLAUDE.md`: `contextual-v1` recorded as the production index profile; `expansion_mode` default unchanged; generation axis noted unverified.
- The `contextual-v1` evaluation artifacts under `eval/reports/`.

### Not done, deliberately

No push, no PR merge, no deploy, no `REFUSAL_COSINE_THRESHOLD` change, no `expansion_mode` default change, no `.env` edit, no corpus change, no `*_v1.0.0*` modification; the one Phase-1 `*_v1.1.0*` artifact touched is `classification_v1.1.0__raw-v1__off.md`, regenerated machine-only from the committed JSONL with identical counts (15/9/2/0). Deployment remains a separate `DEPLOY`-gated manual operation.
