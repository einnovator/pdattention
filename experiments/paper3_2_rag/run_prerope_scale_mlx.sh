#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-docs/papers/shared/results/paper3_2_rag/prerope_causal/scale}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"
SEED="${SEED:-11}"

models=(
  "mlx-community/Qwen3-4B-4bit|qwen3_4b"
  "mlx-community/Qwen3-8B-4bit|qwen3_8b"
  "mlx-community/Llama-3.1-8B-Instruct-4bit|llama3_1_8b"
)

for entry in "${models[@]}"; do
  model="${entry%%|*}"
  slug="${entry##*|}"
  output="$OUTPUT_ROOT/${slug}_seed${SEED}_n${MAX_EXAMPLES}"
  if [[ -s "$output/manifest.json" ]]; then
    echo "[skip] $model already complete"
    continue
  fi
  mkdir -p "$output"
  echo "[run] $model"
  PYTHONPATH=src "$PYTHON_BIN" \
    -m experiments.paper3_2_rag.run_prerope_causal_decomposition \
    --dataset multihoprag \
    --cache-dir .cache/rag_eval \
    --model "$model" \
    --seed "$SEED" \
    --max-examples "$MAX_EXAMPLES" \
    --candidate-count 50 \
    --token-budget 2048 \
    --max-resources 4 \
    --max-new-tokens 16 \
    --output "$output"
done
