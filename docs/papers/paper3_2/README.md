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
