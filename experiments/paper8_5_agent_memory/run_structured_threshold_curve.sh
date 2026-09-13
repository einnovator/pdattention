#!/bin/sh
set -eu

: "${PAPER85_PYTHON:?set PAPER85_PYTHON to the qualified environment Python}"
: "${PAPER85_TOKENIZER:?set PAPER85_TOKENIZER to the pinned tokenizer path}"
: "${PAPER85_OUTPUT:?set PAPER85_OUTPUT to an empty/new output directory}"

PAPER85_MODEL="${PAPER85_MODEL:-qwen3-coder:30b}"
PAPER85_BASE_URL="${PAPER85_BASE_URL:-http://127.0.0.1:11434}"
# Thresholds are task-dependent.  The caller must choose values that cross
# observed tool-result sizes; values inside one materialization equivalence
# class produce the same model-visible requests and are repeated controls, not
# new quality--saving points.
: "${PAPER85_THRESHOLDS:?set PAPER85_THRESHOLDS to task-specific effective breakpoints}"
PAPER85_TRAJECTORY="${PAPER85_TRAJECTORY:-docs/papers/shared/results/paper8_5_agent_memory/frozen_task03_qwen30/trajectory.json}"
PAPER85_REFERENCE="${PAPER85_REFERENCE:-docs/papers/shared/results/paper8_5_agent_memory/frozen_task03_qwen30/full_seed0_a.json}"

mkdir -p "$PAPER85_OUTPUT"
for threshold in $PAPER85_THRESHOLDS; do
  "$PAPER85_PYTHON" -m experiments.paper8_5_agent_memory.run_frozen_replay \
    --trajectory "$PAPER85_TRAJECTORY" \
    --output "$PAPER85_OUTPUT/structured_t${threshold}.json" \
    --base-url "$PAPER85_BASE_URL" \
    --model "$PAPER85_MODEL" \
    --tokenizer "$PAPER85_TOKENIZER" \
    --policy full --head 1 --tail 2 --budget-fraction 1.0 \
    --materialization-mode tool_structured_evidence \
    --materialization-threshold-tokens "$threshold" \
    --seed 0 --timeout 1200 \
    --reference-replay "$PAPER85_REFERENCE" \
    --restart
done
