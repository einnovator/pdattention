# Paper 3.2: RAG, PRA, and Composable Native Memory

Paper 3.2 owns the controlled comparison among full-document context,
conventional RAG, RAG+PRA, and independently composed native memory. It was
branched from Paper 3.1 at commit `24992aeae567b5f5275b9c3e76488f06f524bb1d`.

## Experimental Boundary

The experiment freezes each stage separately:

```text
retrieval -> candidate receipt -> selection -> selection receipt
          -> realization/materialization -> position composition -> generation
```

The canonical RAG+PRA profiles are defined in
`src/pra_hf/rag_composition.py`. A matched Selected Context/Native Memory
comparison must share a selection receipt. A position-policy comparison must
share selected identities, source spans, and token content.

## Inheritance

- Paper 2.5 contributes iterative/associative routing diagnostics.
- Paper 2.6 contributes lexical, semantic, and hybrid discovery channels.
- Papers 2.7-2.9 contribute graph-query and compressed/look-ahead routing arms.
- Paper 3.0 contributes logical-interval materialization, cross-shard gather,
  fixed-budget allocation, evidence-density accounting, and causal controls.
- Paper 3.1 contributes optional summary and multi-index addresses.
- Paper 4.5 contributes the initial RAG dataset harness, immutable receipts,
  powered decomposition, and preliminary model-backed artifacts.
- The former Paper 1.6 plan contributes the multi-resource position-policy
  and D1/D2 permutation protocol.

Inherited Paper 4.5 measurements remain `PRELIMINARY_INHERITED` until they are
reproduced under the Paper 3.2 protocol. See `inheritance_manifest.json`.

## Commands

Focused framework tests:

```bash
PYTHONPATH=src python -m pytest tests/test_rag_composition.py \
  tests/test_rag_retrieval.py tests/test_rag_evaluation.py \
  tests/test_rag_powered.py -q
```

Model-free mechanism smoke:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.run_mechanism_smoke \
  --output docs/papers/shared/results/paper3_2_rag/mechanism_smoke
```

The first model-backed gate uses five separately persisted runs of
`experiments.rag_vs_pra.run_powered_decomposition`, followed by:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.analyze_model_smoke \
  --input-root docs/papers/shared/results/paper3_2_rag/model_smoke \
  --output-dir docs/papers/shared/results/paper3_2_rag/model_smoke/aggregate
```

It is a controlled Qwen3-1.7B-4bit transport smoke over seeds
`11, 23, 37, 71, 101`. Its 150 selected-text/native pairs are output-exact,
but it does not qualify the generic selector or establish natural-task quality.

The first natural replication is under
`docs/papers/shared/results/paper3_2_rag/natural/`. It contains 50 fixed
MultiHop-RAG questions, pinned Qwen3-1.7B-4bit and cross-encoder revisions, 200
exact matched realization pairs, and disjoint cold/warm metrics. This is one
cohort seed and does not replace the later five-seed or cross-family runs.

Regenerate its profile-separated aggregate with:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.analyze_model_smoke \
  --input-root docs/papers/shared/results/paper3_2_rag/natural \
  --run-pattern paper3_2_m1_multihoprag_strong_n20_2k_50 \
  --output-dir docs/papers/shared/results/paper3_2_rag/natural/paper3_2_m1_multihoprag_strong_n20_2k_50/aggregate \
  --selector-profile pra_strong_reranker \
  --scope "50-question MultiHop-RAG cohort; selector-frozen transport and initial natural quality evidence"
```

The completed next-iteration evidence is organized under
`docs/papers/shared/results/paper3_2_rag/`:

- `natural_five_seed/` contains five 30-question natural cohorts and 600 exact
  selected-text/native transport comparisons.
- `local_retrieval/`, `service_retrieval/`, and `service_retrieval_cuda/`
  contain the local and Elasticsearch/Qdrant retrieval ladders.
- `composition_natural/` and `position_natural/` contain the five-seed natural
  materialization, recomposition, repair, order, and position diagnostics.
- `nonprefix_reuse/` contains five seeds of five four-turn changing-selection
  sequences.
- `scale/` contains the reduced 4B, 8B, and cross-family replications.
- `native_records/` contains five 30-question Qwen3-1.7B runs through the
  canonical Paper 4.5 `ContextRecord`/URI/resolver path, their seed aggregate,
  a five-seed Qwen3-4B confirmation, and reduced Qwen3-8B and Llama-3.1-8B
  pilots. Each run keeps
  candidate, reranker, selection, record-resolution, generation, and reuse
  receipts separate.
- `composition_natural/heldout_repair_policy.json` fits a bounded repair
  controller on seeds 11/23/37 and evaluates it without refitting on seeds
  71/101.

First regenerate the seed-aware aggregates from the five raw manifests. The
aggregator retains seed summaries and reports deterministic seed-bootstrap
intervals rather than treating all executions as independent replicates:

```bash
python experiments/paper3_2_rag/aggregate_multiseed.py \
  --kind composition \
  --manifest docs/papers/shared/results/paper3_2_rag/composition_natural/qwen3_1_7b_seed11/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/composition_natural/qwen3_1_7b_seed23/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/composition_natural/qwen3_1_7b_seed37/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/composition_natural/qwen3_1_7b_seed71/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/composition_natural/qwen3_1_7b_seed101/manifest.json \
  --output docs/papers/shared/results/paper3_2_rag/composition_natural/five_seed/manifest.json
```

Build the position aggregate from the same five composition manifests; each
now includes the complete position sweep. Use the same command with
`--kind nonprefix` and the five manifests under `nonprefix_reuse/` to produce
`nonprefix_reuse/five_seed/manifest.json`.
Build the replicated 4B native-record row with:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.aggregate_native_records \
  --manifest docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_seed11/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_seed23/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_seed37/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_seed71/manifest.json \
  --manifest docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_seed101/manifest.json \
  --output docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_five_seed/manifest.json
```

Regenerate the compact publication plots and scale summary with:

```bash
PYTHONPATH=src python -m experiments.paper3_2_rag.build_publication_artifacts \
  --composition-manifest docs/papers/shared/results/paper3_2_rag/composition_natural/five_seed/manifest.json \
  --position-manifest docs/papers/shared/results/paper3_2_rag/position_natural/five_seed/manifest.json \
  --retrieval-summary docs/papers/shared/results/paper3_2_rag/local_retrieval/summary.json \
  --service-summary docs/papers/shared/results/paper3_2_rag/service_retrieval_cuda/summary.json \
  --nonprefix-manifest docs/papers/shared/results/paper3_2_rag/nonprefix_reuse/five_seed/manifest.json \
  --scale-run docs/papers/shared/results/paper3_2_rag/scale/qwen3_4b_seed11 \
  --scale-run docs/papers/shared/results/paper3_2_rag/scale/qwen3_8b_seed11 \
  --scale-run docs/papers/shared/results/paper3_2_rag/scale/llama3_1_8b_seed11 \
  --scale-composition docs/papers/shared/results/paper3_2_rag/scale_composition/qwen3_4b_seed11 \
  --scale-composition docs/papers/shared/results/paper3_2_rag/scale_composition/qwen3_8b_seed11 \
  --scale-composition docs/papers/shared/results/paper3_2_rag/scale_composition/llama3_1_8b_seed11 \
  --native-record-aggregate docs/papers/shared/results/paper3_2_rag/native_records/five_seed/manifest.json \
  --native-record-scale-aggregate docs/papers/shared/results/paper3_2_rag/native_records/qwen3_4b_five_seed/manifest.json \
  --native-record-scale-run docs/papers/shared/results/paper3_2_rag/native_records/qwen3_8b_seed11 \
  --native-record-scale-run docs/papers/shared/results/paper3_2_rag/native_records/llama3_1_8b_seed11 \
  --heldout-repair-policy docs/papers/shared/results/paper3_2_rag/composition_natural/heldout_repair_policy.json \
  --output-dir docs/papers/shared/results/paper3_2_rag/publication
```
