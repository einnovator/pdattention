# Paper 3.2 experiments

These runners keep retrieval, selection, realization, and positional
composition as separate factors. Every model-backed run pins model revisions
and writes row-level receipts plus an aggregate manifest.

## Natural five-seed transport

`run_powered_decomposition` compares selected text, contiguous native K/V, and
ordinary prefix-cache reuse from the same selection receipt. Use five seeds as
the replication units and aggregate them with `analyze_model_smoke`.

## Composition

`run_composition_fidelity` compares fresh packed encoding, contiguous native
K/V, independently encoded source-local K/V, GLOBAL_PACKED RoPE rebinding, and
a diagnostic repair curve. Repair rows use already computed fresh states and
must not be presented as deployable compute savings.

## Retrieval

`run_retrieval_ladder` measures local BM25, BGE dense FAISS, hybrid RRF, the
historical Paper-4.5 MiniLM reranker, and the stronger BGE-v2-M3 reranker.

For service-backed retrieval:

```bash
docker compose -f experiments/paper3_2_rag/docker-compose.retrieval.yml up -d
python -m experiments.paper3_2_rag.run_service_retrieval \
  --rebuild \
  --index-revision multihoprag-bb98345-paper32-v1 \
  --output /tmp/paper3_2_service_retrieval
```

The service run records index build cost and search p50/p95/p99 separately
from model inference. Stop the services with:

```bash
docker compose -f experiments/paper3_2_rag/docker-compose.retrieval.yml down
```
