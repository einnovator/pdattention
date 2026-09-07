# Paper 3.3 cross-document retrieval expansion

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
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/validation_n150
```

The output records support-span/chunk recovery, distractor fraction, requested
and deduplicated expansion tokens, already-resident versus newly materialized
native tokens, search latency, and complete policy receipts. These are
retrieval/mechanism measurements. They are not answer F1 or an SDK promotion
until the generation, sparse-SA, five-seed, scale, and family gates pass.
