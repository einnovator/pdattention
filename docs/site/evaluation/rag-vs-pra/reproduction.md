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

## Analyze

```bash
python -m experiments.rag_vs_pra.analyze_eval_ladder \
  results/multihoprag_l1.json.gz \
  results/multihoprag_l2.json.gz \
  --output-dir results/summary
```

Committed detailed artifacts are gzip-compressed JSON. Their receipts, source revisions, questions, gold evidence, selected spans, metrics, and failure classes remain directly inspectable after decompression.
