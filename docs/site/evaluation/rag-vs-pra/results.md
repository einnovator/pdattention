# Results

## MultiHop-RAG selection probe

The table reports the 2K physical-budget result over 50 held-out questions. `Answer present` means an accepted answer string occurs in selected context; it is not generated-answer EM.

| Stage | Candidates | Condition | Answer present | Support docs | Support spans | False selected docs |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Gold-present L1 | 5 | Standard RAG | 0.64 | 0.670 | 0.368 | 0.518 |
| Gold-present L1 | 5 | RAG + PRA | **0.74** | **0.818** | **0.525** | **0.468** |
| Gold-present L1 | 20 | Standard RAG | 0.60 | 0.508 | 0.315 | 0.736 |
| Gold-present L1 | 20 | RAG + PRA | **0.70** | **0.618** | **0.385** | **0.721** |
| Gold-present L1 | 50 | Standard RAG | 0.64 | 0.522 | **0.345** | **0.774** |
| Gold-present L1 | 50 | RAG + PRA | **0.76** | **0.538** | 0.310 | 0.785 |
| Real BM25 L2 | 5 | Standard RAG | 0.58 | 0.405 | 0.247 | **0.684** |
| Real BM25 L2 | 5 | RAG + PRA | **0.68** | **0.472** | **0.327** | 0.701 |
| Real BM25 L2 | 20 | Standard RAG | 0.56 | 0.430 | 0.268 | **0.769** |
| Real BM25 L2 | 20 | RAG + PRA | **0.68** | **0.493** | **0.318** | 0.777 |
| Real BM25 L2 | 50 | Standard RAG | 0.64 | **0.515** | **0.338** | **0.776** |
| Real BM25 L2 | 50 | RAG + PRA | **0.76** | 0.505 | 0.300 | 0.795 |

At 2K, generic PRA improves answer availability by 0.10--0.12 absolute across these displayed settings. The cleanest mechanism result is at 5--20 candidates, where supporting-document and span coverage also improve. At 50 real-retrieval candidates, answer availability rises while gold coverage falls slightly and false selection increases. That row is a warning against interpreting answer-string presence as grounded answer quality.

At 8K--16K, the selectors converge because most useful chunks fit. The measured opportunity is therefore budget pressure, not an unconditional quality advantage.

![MultiHop-RAG candidate and budget curves](../../assets/results/rag/multihoprag_fixed_candidate_curves.png)

## Reuse diagnostic

![MultiHop-RAG cumulative reuse curve](../../assets/results/rag/multihoprag_reuse_curve.png)

The current persistent-corpus cohort avoids 20.8% of repeated chunk materialization after 50 queries. It does not yet include engine-measured TTFT or physical native-cache residency.

## Qwen3-4B MLX generation smoke

Ten of the same L1 questions were executed with the exact `mlx-community/Qwen3-4B-4bit` revision on a 48 GiB M4 Pro. Standard RAG used selected text in the ordinary prompt cache. PRA used the generic selector and all-layer detached native K/V. Both used greedy 32-token generation after symmetric path warm-up.

| Candidates | Condition | Official task score | Strict EM | Token F1 | TTFT ms | Completion ms | Visible prompt tokens | Native K/V tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Standard RAG | **0.90** | 0.00 | **0.094** | 172.2 | **563.4** | 2,110 | 0 |
| 20 | RAG + PRA | 0.70 | 0.00 | 0.077 | 172.8 | 646.6 | **82** | 2,032 |
| 50 | Standard RAG | **0.90** | 0.00 | **0.091** | **165.4** | **557.1** | 2,107 | 0 |
| 50 | RAG + PRA | 0.80 | 0.00 | 0.081 | 174.5 | 646.9 | **82** | 2,035 |

The model often generated explanatory sentences, so strict short-answer EM is zero even when the gold answer token appears. The official task score is therefore more informative here, but the strict values remain visible. PRA reduces visible prompt tokens by about 96%, yet its all-layer 2K-token native memory occupies about 300 MB. Cold source ingestion is similar (`2.39--2.46` seconds), while native completion is about 15% slower. This smoke closes neither the quality nor economic gate.

The product comparison changes both context selection and representation, by design. It does not by itself attribute the quality loss to routing versus native transport. Prior frozen-selection MLX parity checks cover transport; a larger RAG cohort should repeat that control within this exact benchmark.

## Claim boundary

The 50-question curves establish auditable candidate controls and a parameter-free selection signal. The 10-question model smoke establishes a negative end-to-end gate, not superiority. A learned-adaptor gain and engine-level economic win remain unestablished. Those claims require a larger model-backed cohort, frozen selected-text/native-K/V transport inside this workload, and repeated-query engine measurements.
