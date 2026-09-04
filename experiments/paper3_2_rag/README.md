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

## Native record references and internal reranking

`run_native_record_reranker` exercises the high-level PRA document path. It
ingests typed document and chunk records, binds lightweight `<REF_n>` prompt
tokens to stable `pra://` URIs, resolves explicit or routed-root references,
and only then materializes independently cached chunk K/V. The same frozen
selection is also run as packed text and one contiguous native block.

The selector matrix places MiniLM or BGE-v2-M3 after a bounded BM25 chunk
cohort. Every reranker model, revision, input rank, score, output rank, and
latency is retained in a receipt. The default run also compares packed and
record-relative order sensitivity and constructs changing-query sequences for
record-level reuse accounting.

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.run_native_record_reranker \
  --model mlx-community/Qwen3-1.7B-4bit \
  --seed 11 \
  --max-examples 30 \
  --selectors bm25,minilm,bge \
  --reranker-device cpu \
  --output reports/paper3_2_native_records/qwen3_1_7b_seed11
```

The logical reference prompt is retained for audit, but reference-control
tokens are consumed by the runtime and are not serialized as evidence text to
an unadapted base model. This keeps persistent record identity separate from
the model-visible question and from the materialized native K/V.

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
