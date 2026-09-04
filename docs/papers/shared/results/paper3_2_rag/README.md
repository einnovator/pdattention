# Paper 3.2 result artifacts

This directory contains the committed evidence used by Paper 3.2.

- `natural_five_seed/` contains five Qwen3-1.7B MultiHop-RAG runs for seeds
  11, 23, 37, 71, and 101, plus an aggregate generated from their raw rows.
- `service_retrieval/` contains the 50-question Elasticsearch and Qdrant
  retrieval run with query embedding, service search, and client overhead
  reported separately.
- `service_retrieval_cuda/` repeats the same frozen 50-question service run
  with BGE query embedding on an NVIDIA RTX 5060. The retrieval identities and
  recall values are unchanged; the artifact isolates the embedding-device
  effect from Elasticsearch/Qdrant service time.
- `local_retrieval/` contains the matched 50-question CUDA retrieval ladder:
  BM25, FAISS/BGE dense, hybrid RRF, the historical MiniLM reranker, and the
  stronger BGE-v2-M3 reranker.
- `composition_natural/` contains selector-frozen natural partial-
  materialization, independent-composition, repair-geometry, and document-order
  measurements. Repair rows are diagnostic because fresh packed states were
  computed before replacement.
- `nonprefix_reuse/` contains natural changing-selection sequences. It compares
  fresh reprefill, ordinary exact-prefix reuse, contiguous selection reuse,
  independently cached resource reuse, rebinding plus diagnostic repair, and
  partial materialization while retaining per-turn selection receipts.
- `position_natural/` contains the natural positional-policy sweep over
  source-local, packed rebound, adjacent, rank, score, near-band, and random
  geometries.
- `scale/` contains the reduced model-scaling replications. Each run retains
  candidate and selection receipts, compressed per-condition rows, and a
  cohort manifest; `publication/publication_summary.json` regenerates their
  parity and timing summaries.
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
