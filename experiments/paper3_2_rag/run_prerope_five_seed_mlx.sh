#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-docs/papers/shared/results/paper3_2_rag/prerope_causal}"
MODEL="${MODEL:-mlx-community/Qwen3-1.7B-4bit}"
SEEDS="${SEEDS:-11 23 37 71 101}"
WAIT_PID="${WAIT_PID:-}"

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 15
  done
fi

for seed in $SEEDS; do
  output="$OUTPUT_ROOT/qwen3_1_7b_seed${seed}"
  if [[ -s "$output/manifest.json" ]]; then
    echo "[skip] seed $seed already complete"
    continue
  fi
  mkdir -p "$output"
  echo "[run] seed $seed"
  PYTHONPATH=src "$PYTHON_BIN" \
    -m experiments.paper3_2_rag.run_prerope_causal_decomposition \
    --dataset multihoprag \
    --cache-dir .cache/rag_eval \
    --model "$MODEL" \
    --seed "$seed" \
    --max-examples 30 \
    --candidate-count 50 \
    --token-budget 2048 \
    --max-resources 4 \
    --max-new-tokens 16 \
    --output "$output"
done

