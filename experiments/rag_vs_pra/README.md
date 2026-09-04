# Powered RAG decomposition

This experiment separates first-stage retrieval, context selection, execution
representation, and adapter effects on a fixed MultiHop-RAG cohort.

The normalized condition IDs are:

- `NO_PRA_STANDARD_RAG`
- `PRA_SELECTED_CONTEXT_NO_ADAPTOR`
- `PRA_NATIVE_MEMORY_NO_ADAPTOR`
- `PRA_SELECTED_CONTEXT_BUNDLE`
- `PRA_NATIVE_MEMORY_BUNDLE`

Bundle rows remain `NO_QUALIFIED_ADAPTER` until the exact model, precision, and
MultiHop-RAG profile have a qualified adapter. Selected Context and Native
Memory rows share an immutable `SelectionReceipt`; the run aborts if selected
documents, chunks, intervals, scores, order, or budget differ.

## Contract smoke

```bash
PYTHONPATH=src python -m experiments.rag_vs_pra.run_powered_decomposition \
  --dataset fixture \
  --max-examples 3 \
  --candidate-counts 20 \
  --token-budgets 32 \
  --backend probe \
  --skip-strong \
  --output out/rag-powered-smoke

PYTHONPATH=src python -m experiments.rag_vs_pra.analyze_powered_decomposition \
  --input-dir out/rag-powered-smoke \
  --primary-candidate-count 20 \
  --primary-token-budget 32 \
  --minimum-examples 3
```

## Primary powered cell

```bash
PYTHONPATH=src python -m experiments.rag_vs_pra.run_powered_decomposition \
  --dataset multihoprag \
  --max-examples 50 \
  --candidate-counts 20 \
  --token-budgets 2048 \
  --backend mlx-native \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --max-new-tokens 32 \
  --output out/rag-powered-qwen3-4b
```

The runner resolves and stores an immutable reranker revision. The default is
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Use `--skip-strong` only for plumbing
smokes; it is not a powered qualification configuration.

For physical persistent-corpus reuse, run the same frozen generic-PRA selection
as visible text and as chunk-resident native K/V. The native cache survives
across questions and reports physical chunk hits and retained bytes:

```bash
PYTHONPATH=src python -m experiments.rag_vs_pra.run_powered_decomposition \
  --dataset multihoprag --max-examples 50 --seed 11 \
  --candidate-counts 20 --token-budgets 2048 \
  --backend mlx-native \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --regimes persistent-corpus --native-cache-unit chunk \
  --skip-strong \
  --output out/rag-powered-qwen3-4b-persistent
```

## Artifacts

The runner writes the cohort manifest, candidate receipts, selection receipts,
and compressed condition rows. The analyzer adds condition and failure
summaries, matched deltas, persistent-corpus curves, qualification gates, a
paper table, an HF-card fragment, and plots. Cold and warm rows remain separate.
