# Paper 3.2 result artifacts

This directory contains the committed evidence used by Paper 3.2.

- `natural_five_seed/` contains five Qwen3-1.7B MultiHop-RAG runs for seeds
  11, 23, 37, 71, and 101, plus an aggregate generated from their raw rows.
- `service_retrieval/` contains the 50-question Elasticsearch and Qdrant
  retrieval run with query embedding, service search, and client overhead
  reported separately.
- `local_retrieval/` contains the matched 50-question CUDA retrieval ladder:
  BM25, FAISS/BGE dense, hybrid RRF, the historical MiniLM reranker, and the
  stronger BGE-v2-M3 reranker.
- `mechanism_smoke/` and `model_smoke/` contain earlier implementation gates.

Regenerate the five-seed aggregate from the repository root with:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.paper3_2_rag.analyze_model_smoke `
  --input-root docs/papers/shared/results/paper3_2_rag/natural_five_seed `
  --output-dir docs/papers/shared/results/paper3_2_rag/natural_five_seed/aggregate `
  --run-pattern "paper3_2_m1_multihoprag_seed*" `
  --selector-profile pra_strong_reranker `
  --scope "five seed MultiHop-RAG natural cohorts; 30 examples per seed"
```

Regeneration on Python 3.10 reproduces all counts, means, parity results, and
ratios. A few seed-standard-deviation values can differ at the final binary
floating-point digit from the aggregate generated on macOS Python 3.12.

The service retrieval latency is intentionally decomposed. In the committed
run, Qdrant search takes milliseconds while CPU BGE query embedding takes about
one second. Do not present their sum as database search latency.
