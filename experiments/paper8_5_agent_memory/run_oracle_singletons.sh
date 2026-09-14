#!/bin/sh
set -eu

# Run one frozen decision with one complete middle causal group omitted at a
# time. The reference action remains in the scorer artifact and is never added
# to the model request. This is oracle evidence, not a deployable selector.
: "${PAPER85_PYTHON:?set PAPER85_PYTHON to the qualified environment Python}"
: "${PAPER85_TOKENIZER:?set PAPER85_TOKENIZER to the pinned tokenizer path}"
: "${PAPER85_TRAJECTORY:?set PAPER85_TRAJECTORY to the canonical trajectory}"
: "${PAPER85_REFERENCE:?set PAPER85_REFERENCE to repeat-qualified FULL-A}"
: "${PAPER85_DECISION:?set PAPER85_DECISION to the one-based frozen decision}"
: "${PAPER85_LAST_GROUP:?set PAPER85_LAST_GROUP to the last middle turn index}"
: "${PAPER85_OUTPUT:?set PAPER85_OUTPUT to the singleton output directory}"

PAPER85_MODEL="${PAPER85_MODEL:-qwen3-coder:30b}"
PAPER85_BASE_URL="${PAPER85_BASE_URL:-http://127.0.0.1:11434}"
PAPER85_FIRST_GROUP="${PAPER85_FIRST_GROUP:-1}"
PAPER85_HEAD="${PAPER85_HEAD:-1}"
PAPER85_TAIL="${PAPER85_TAIL:-2}"

mkdir -p "$PAPER85_OUTPUT"
index="$PAPER85_FIRST_GROUP"
while [ "$index" -le "$PAPER85_LAST_GROUP" ]; do
  group=$(printf 'turn:t%04d' "$index")
  label=$(printf 't%04d' "$index")
  echo "START $label $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  "$PAPER85_PYTHON" -m experiments.paper8_5_agent_memory.run_frozen_replay \
    --trajectory "$PAPER85_TRAJECTORY" \
    --output "$PAPER85_OUTPUT/single_${label}.json" \
    --base-url "$PAPER85_BASE_URL" \
    --model "$PAPER85_MODEL" \
    --tokenizer "$PAPER85_TOKENIZER" \
    --policy full \
    --head "$PAPER85_HEAD" \
    --tail "$PAPER85_TAIL" \
    --budget-fraction 1.0 \
    --min-decision "$PAPER85_DECISION" \
    --max-decisions 1 \
    --max-output-tokens 1024 \
    --seed 0 \
    --timeout 1200 \
    --reference-replay "$PAPER85_REFERENCE" \
    --oracle-omit-group "$group"
  index=$((index + 1))
done

echo "COMPLETE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
