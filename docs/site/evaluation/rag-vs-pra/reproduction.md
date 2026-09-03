# Reproduction

Install the research dependencies and run from the repository root:

```bash
pip install -e ".[research]"
```

## Controlled fixture

```bash
python -m experiments.rag_vs_pra.run_eval_ladder \
  --dataset fixture --stage fixed \
  --candidate-counts 5,10,20,50 \
  --token-budgets 32,64,128,256 \
  --max-examples 15 \
  --output results/fixture_l0.json.gz
```

## MultiHop-RAG fixed candidates

The loader downloads the two official release files and records their digests.

```bash
python -m experiments.rag_vs_pra.run_eval_ladder \
  --dataset multihoprag --stage fixed \
  --candidate-counts 5,10,20,50 \
  --token-budgets 2048,4096,8192,16384 \
  --max-examples 50 --seed 11 \
  --output results/multihoprag_l1.json.gz
```

Change `--stage fixed` to `--stage retrieval` for unmodified BM25 candidate sets.

## Native MLX generation

```bash
python -m experiments.rag_vs_pra.run_eval_ladder \
  --dataset multihoprag --stage fixed \
  --candidate-counts 20,50 --token-budgets 2048 \
  --max-examples 10 --seed 11 \
  --backend mlx-native \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --max-new-tokens 32 \
  --output results/multihoprag_mlx_native.json.gz
```

## Native HF generation

```bash
python -m experiments.rag_vs_pra.run_eval_ladder \
  --dataset multihoprag --stage fixed \
  --candidate-counts 20,50 --token-budgets 2048 \
  --max-examples 50 --seed 11 \
  --backend hf-native \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --device mps --consumption-layers 24 \
  --output results/multihoprag_hf_native.json.gz
```

The native backend warms visible and detached-K/V paths before recording rows. Use all eligible consumer layers for correctness qualification; reduced profiles require separate calibration.

## Powered decomposition

The powered runner keeps real BM25 candidates, emits separate Selected Context
and Native Memory rows, and pins the strong cross-encoder revision. Cold and
warm rows are not averaged.

```bash
python -m experiments.rag_vs_pra.run_powered_decomposition \
  --dataset multihoprag \
  --max-examples 50 --seed 11 \
  --candidate-counts 20 --token-budgets 2048 \
  --backend mlx-native \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --max-new-tokens 32 \
  --output results/rag-powered-qwen3-4b

python -m experiments.rag_vs_pra.analyze_powered_decomposition \
  --input-dir results/rag-powered-qwen3-4b \
  --primary-candidate-count 20 \
  --primary-token-budget 2048 \
  --minimum-examples 50
```

The output directory contains `cohort_manifest.json`, candidate and selection
receipts, `condition_results.jsonl.gz`, summaries, matched deltas, failure
counts, qualification gates, paper/card fragments, and plots. Bundle conditions
are emitted as `NO_QUALIFIED_ADAPTER` rather than borrowed from another task or
precision.

## Analyze

```bash
python -m experiments.rag_vs_pra.analyze_eval_ladder \
  results/multihoprag_l1.json.gz \
  results/multihoprag_l2.json.gz \
  --output-dir results/summary
```

Committed detailed artifacts are gzip-compressed JSON. Their receipts, source revisions, questions, gold evidence, selected spans, metrics, and failure classes remain directly inspectable after decompression.
