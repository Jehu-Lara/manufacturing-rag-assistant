# Refusal Threshold Analysis — eval_set v1.0.0

Computed from real hybrid-retrieval runs over the hash-verified eval set. Scores are each question's top-1 *pure-semantic* cosine similarity (`retrieval.hybrid.RetrievalResult.semantic_score` at `semantic_rank == 1`), not the RRF `fused_score` — see SPEC.md's Phase 2 status for why `fused_score` is disqualified as a refusal-confidence signal (rank-based, not magnitude-based).

## Unanswerable subset — top-1 semantic_score (n=10)

- Sorted: [0.4799048900604248, 0.4941929578781128, 0.4966537356376648, 0.5146251320838928, 0.5578610301017761, 0.590476930141449, 0.5967954993247986, 0.6260544061660767, 0.6346091032028198, 0.6547982692718506]
- Stats: min=0.4799, max=0.6548, mean=0.5646, median=0.5742

## Answerable subset — top-1 semantic_score (n=30)

- Sorted: [0.5796104669570923, 0.583651602268219, 0.6027761697769165, 0.6061258316040039, 0.6123687028884888, 0.6134135127067566, 0.6360657811164856, 0.6401943564414978, 0.6617925763130188, 0.6722861528396606, 0.679008424282074, 0.6818394064903259, 0.6854784488677979, 0.689324676990509, 0.6940515041351318, 0.6946896910667419, 0.697928249835968, 0.6983100175857544, 0.711088240146637, 0.7115805149078369, 0.715408205986023, 0.7167966365814209, 0.7226194739341736, 0.7300907373428345, 0.7314331531524658, 0.7397452592849731, 0.7431122064590454, 0.7440525889396667, 0.765110969543457, 0.829218864440918]
- Stats: min=0.5796, max=0.8292, mean=0.6863, median=0.6944

## Cutoff sweep

Refusal rule under test at each candidate threshold: refuse when top-1 semantic_score < threshold.

| threshold | answerable wrongly refused | unanswerable correctly refused | objective (correct - wrong) |
|---|---|---|---|
| 0.4599 | 0 | 0 | 0 |
| 0.4799 | 0 | 1 | 1 |
| 0.4999 | 0 | 3 | 3 |
| 0.5199 | 0 | 4 | 4 |
| 0.5399 | 0 | 4 | 4 |
| 0.5599 | 0 | 5 | 5 |
| 0.5799 | 1 | 5 | 4 |
| 0.5999 | 2 | 7 | 5 |
| 0.6199 | 6 | 7 | 1 |
| 0.6399 | 7 | 9 | 2 |
| 0.6599 | 8 | 10 | 2 |
| 0.6799 | 11 | 10 | -1 |
| 0.6999 | 18 | 10 | -8 |
| 0.7199 | 22 | 10 | -12 |
| 0.7399 | 26 | 10 | -16 |
| 0.7599 | 28 | 10 | -18 |
| 0.7799 | 29 | 10 | -19 |
| 0.7999 | 29 | 10 | -19 |
| 0.8199 | 29 | 10 | -19 |
| 0.8399 | 30 | 10 | -20 |

## Selection procedure and result

- **Branch taken: overlap.** max(unanswerable top-1 semantic_score) = 0.6548 >= min(answerable top-1 semantic_score) = 0.5796 — the two ranges overlap.
- Rule applied: pick the sweep-table threshold that maximizes (unanswerable_correctly_refused - answerable_wrongly_refused); ties broken toward the lowest candidate threshold.
- Chosen threshold = **0.5599**.

**Chosen REFUSAL_COSINE_THRESHOLD: 0.5599**

