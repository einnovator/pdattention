# Paper 3.2 experiments

These runners keep retrieval, selection, realization, and positional
composition as separate factors. Every model-backed run pins model revisions
and writes row-level receipts plus an aggregate manifest.

When multiple worktrees have editable PRA installs, set `PYTHONPATH` to this
worktree's `src` directory before running experiments.

## Fast mechanism smoke

The dependency-light first gate validates candidate/selection receipts and the
RAG+PRA position/profile matrix over five synthetic corpus seeds. It does not
load a language model and therefore makes no answer-quality claim.

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
python -m experiments.paper3_2_rag.run_mechanism_smoke
```

Its canonical artifacts are under
`docs/papers/shared/results/paper3_2_rag/mechanism_smoke/`.

## Natural five-seed transport

`run_powered_decomposition` compares selected text, contiguous native K/V, and
ordinary prefix-cache reuse from the same selection receipt. Use five seeds as
the replication units and aggregate them with `analyze_model_smoke`.

## Composition

`run_composition_fidelity` compares fresh packed encoding, contiguous native
K/V, independently encoded source-local K/V, GLOBAL_PACKED RoPE rebinding, and
a diagnostic repair curve. Repair rows use already computed fresh states and
must not be presented as deployable compute savings.

The next-iteration controls add exact token-budget partial materialization,
evidence-aware oracle and equal-budget wrong-memory conditions, deterministic
prefix/boundary/later-resource repair geometries, first-step JS/KL diagnostics,
and optional receipt-driven position policies.

## Changing-selection reuse

`run_nonprefix_reuse` builds deterministic natural query sequences whose frozen
strong-reranker selections overlap. It measures ordinary exact-prefix reuse,
exact contiguous-block reuse, persistent chunk-native source-local and rebound
composition, boundary repair, and partial materialization with disjoint token
accounting.

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.run_nonprefix_reuse \
  --cache-dir .cache/rag_eval \
  --candidate-questions 50 \
  --sequence-length 4 \
  --sequence-count 5 \
  --repair-fraction 0.25 \
  --partial-fraction 0.5 \
  --output reports/paper3_2_nonprefix_reuse
```

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
