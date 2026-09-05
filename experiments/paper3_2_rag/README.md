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

## Pre-RoPE causal decomposition

`run_prerope_causal_decomposition` freezes BM25 candidates, BGE-v2-M3
ranking, selected records, separators, order, and exact packed positions. It
then compares ordinary packed causal RAG (A), the same packed tokens with
document-to-document attention blocked (B), and independently encoded
pre-RoPE records rebound to B's exact request positions (C). B/C layerwise K/V,
final-logit, NLL, and output diagnostics are correctness gates. The additional
mask ladder tests previous-document, top-ranked-document, and 8/16/32/64-token
boundary interaction without changing retrieval.

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.run_prerope_causal_decomposition \
  --dataset multihoprag \
  --model mlx-community/Qwen3-1.7B-4bit \
  --seed 11 \
  --max-examples 30 \
  --candidate-count 50 \
  --token-budget 2048 \
  --max-resources 4 \
  --output reports/paper3_2_prerope/qwen3_1_7b_seed11
```

The pre-RoPE cache stores host-projected keys before rotation plus unchanged
values and a model/revision/frequency/layout contract. Materialization fails
if that contract does not match the host. Post-RoPE storage remains the default
for fixed-position memories.

Aggregate the five independently selected cohorts with the seed, rather than
the individual question, as the replication unit:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.aggregate_prerope_causal \
  --manifest reports/paper3_2_prerope/qwen3_1_7b_seed11/manifest.json \
  --manifest reports/paper3_2_prerope/qwen3_1_7b_seed23/manifest.json \
  --manifest reports/paper3_2_prerope/qwen3_1_7b_seed37/manifest.json \
  --manifest reports/paper3_2_prerope/qwen3_1_7b_seed71/manifest.json \
  --manifest reports/paper3_2_prerope/qwen3_1_7b_seed101/manifest.json \
  --output reports/paper3_2_prerope/five_seed
```

When the five-seed A/B/C comparison is informative, run the preregistered
reduced scale check (ten questions each, one seed) serially on the same host:

```bash
PYTHON_BIN=/path/to/mlx/python \
  bash experiments/paper3_2_rag/run_prerope_scale_mlx.sh
```

The scale check retains the full mask ladder. It is a model-family replication,
not a replacement for the five-seed Qwen3-1.7B estimate.

## Native record references and internal reranking

`run_native_record_reranker` exercises the high-level PRA document path. It
ingests typed document and chunk records, binds lightweight `<REF_n>` prompt
tokens to stable `pra://` URIs, resolves explicit or routed-root references,
and only then materializes independently cached chunk K/V. The same frozen
selection is also run as packed text and one contiguous native block.

The records are the shared Paper 4.5 `ContextRecord` objects, not a parallel
experiment schema. Documents use the canonical generic-document payload and
policy; chunks use `RAG_CHUNK` children with parent identity and source-relative
offsets. Ingestion receipts mirror the semantic fields consumed by the Paper
4.5 storage lifecycle. Model/layout compatibility remains an engine-time
storage fingerprint rather than persistent source metadata.

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

Aggregate replicated scale runs separately from one-seed pilots so their
replication unit and confidence intervals remain visible:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.aggregate_native_records \
  --manifest reports/paper3_2_native_records/qwen3_4b_seed11/manifest.json \
  --manifest reports/paper3_2_native_records/qwen3_4b_seed23/manifest.json \
  --manifest reports/paper3_2_native_records/qwen3_4b_seed37/manifest.json \
  --manifest reports/paper3_2_native_records/qwen3_4b_seed71/manifest.json \
  --manifest reports/paper3_2_native_records/qwen3_4b_seed101/manifest.json \
  --output reports/paper3_2_native_records/qwen3_4b_five_seed/manifest.json
```

`calibrate_composition_policy` fits an intentionally small repair controller
on calibration seeds and evaluates the frozen actions on disjoint seeds. The
controller sees question type, native-token length, and resource count only;
answers, support labels, and evaluation outcomes are unavailable at decision
time. Its objective combines first-step JS divergence with recomputation cost.
This is a composition-fidelity diagnostic, not a learned answer-quality claim.

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.calibrate_composition_policy \
  --calibration-manifest reports/composition/qwen3_1_7b_seed11/manifest.json \
  --calibration-manifest reports/composition/qwen3_1_7b_seed23/manifest.json \
  --calibration-manifest reports/composition/qwen3_1_7b_seed37/manifest.json \
  --evaluation-manifest reports/composition/qwen3_1_7b_seed71/manifest.json \
  --evaluation-manifest reports/composition/qwen3_1_7b_seed101/manifest.json \
  --output reports/composition/heldout_repair_policy.json
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
