# Paper 3.3 cross-document retrieval expansion

Install the research dependencies with `pip install -e '.[research]'`. Natural
runs use deterministic BM25-v2 receipts; v2 counts document occurrence for IDF
and sorts query terms before floating-point accumulation.

This experiment keeps first-stage retrieval, reranking, selected records, and
initial selected spans frozen. A selected span may search only peer selected
records. The runtime then enforces authorization, interval deduplication, and a
global extra-token budget before emitting source-addressed materialization
spans and a hash-linked receipt.

Run the correctness fixture:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_frontier \
  --dataset fixture --max-examples 5 --candidate-count 25 \
  --chunk-tokens 32 --chunk-overlap 4 --max-resources 2 \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/fixture_smoke
```

Run the frozen natural validation cohort with the inherited strong reranker:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_frontier \
  --dataset multihoprag --split-name validation --max-examples 150 \
  --selector cross_encoder --reranker BAAI/bge-reranker-v2-m3 \
  --dense-model BAAI/bge-base-en-v1.5 \
  --selection-cache .runs/paper3_3_validation_selection.jsonl \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/validation_n150
```

The dense and RRF arms use the pinned sentence-transformer's distinct query and
document encodings. The runtime retains a dependency-free signed-hash fallback,
but powered semantic results must identify the model and immutable revision in
their summary. The output records support-span/chunk recovery, distractor fraction, requested
and deduplicated expansion tokens, already-resident versus newly materialized
native tokens, search latency, and complete policy receipts. These are
retrieval/mechanism measurements. They are not answer F1 or an SDK promotion
until the generation, sparse-SA, five-seed, scale, and family gates pass.

After selecting a policy and budget on validation, replay that frozen choice
through the native MLX path:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_generation \
  --cache-dir .cache/rag_eval --split-name test --max-examples 150 \
  --model mlx-community/Qwen3-1.7B-4bit \
  --mode dense --no-query-conditioned --top-k 1 \
  --cross-token-budget 64 \
  --selection-cache .runs/paper3_3_test_selection.jsonl \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/generation_test_qwen17b
```

The JSONL cache freezes tokenizer-independent source intervals and validates
the candidate receipt, selector revision, token budget, and resource limit on
every replay. Model-specific native token counts are measured only when those
fixed intervals are encoded. This keeps policy and cross-model comparisons
paired while avoiding repeated cross-encoder inference.

The dense selected-only, k=1, 64-token policy above is fixed by the corrected
BM25-v2 validation frontier. Do not replace it using test retrieval or answer
metrics. Summarize a held-out retrieval replay with `summarize_expansion
--frozen-config <validation-publication-summary>` and a generation run with
`summarize_generation`; both commands preserve the cache digest and distinguish
bootstrap resampling seeds from independent model trials.

The generation runner reports eight separately named conditions. Four form an
explicit expansion-by-interaction factorial:

| Condition | Added evidence | Cross-record interaction |
| --- | --- | --- |
| `INDEPENDENT_PRA` | no | no |
| `PAIR_SA_ONLY` | no | retrieval-linked original record pairs |
| `EXPANSION_ONLY` | yes | no |
| `CROSSDOC_EXPANSION_PAIR_SA` | yes | retrieval-linked pairs |

Packed RAG is the dense causal reference. `PAIR_SA_ONLY_BOUNDARY` and
`CROSSDOC_EXPANSION_PAIR_SA_BOUNDARY` are matched boundary-token controls, and
the linked-region physical-edge oracle is a mechanism control. The interaction
conditions reuse Paper 3.3's host attention instrumentation and are not
sparse-kernel speed measurements.

Run the same frozen test identities at both audit budgets by changing only
`--token-budget 512` to `--token-budget 1024`. Each output row records
`source_token_budget` and immutable selection and expansion receipt identities.
