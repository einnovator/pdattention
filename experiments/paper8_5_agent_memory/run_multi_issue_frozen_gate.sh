#!/usr/bin/env sh
set -eu

: "${PAPER85_PYTHON:?set PAPER85_PYTHON}"
: "${PAPER85_BASE_URL:?set PAPER85_BASE_URL}"
: "${PAPER85_MODEL:?set PAPER85_MODEL}"
: "${PAPER85_TOKENIZER:?set PAPER85_TOKENIZER}"
: "${PAPER85_OUTPUT:?set PAPER85_OUTPUT}"

PAPER85_REPO=${PAPER85_REPO:-$(pwd)}
PAPER85_MAX_DECISIONS=${PAPER85_MAX_DECISIONS:-1}
PAPER85_MAX_OUTPUT_TOKENS=${PAPER85_MAX_OUTPUT_TOKENS:-1024}
PAPER85_SEED=${PAPER85_SEED:-0}
PAPER85_ARTIFACT_ROOT="$PAPER85_REPO/docs/papers/shared/results/paper8_5_agent_memory/multi_issue_session_v1"

mkdir -p "$PAPER85_OUTPUT/session_2" "$PAPER85_OUTPUT/session_3"

run_replay() {
  session_count=$1
  first_decision=$2
  policy=$3
  output=$4
  reference=${5:-}
  trajectory="$PAPER85_ARTIFACT_ROOT/session_${session_count}_issues.json"
  set -- "$PAPER85_PYTHON" -m experiments.paper8_5_agent_memory.run_frozen_replay \
    --trajectory "$trajectory" --output "$output" \
    --base-url "$PAPER85_BASE_URL" --model "$PAPER85_MODEL" \
    --tokenizer "$PAPER85_TOKENIZER" --policy "$policy" \
    --budget-fraction 1.0 --min-decision "$first_decision" \
    --max-decisions "$PAPER85_MAX_DECISIONS" \
    --max-output-tokens "$PAPER85_MAX_OUTPUT_TOKENS" --seed "$PAPER85_SEED"
  if [ -n "$reference" ]; then
    set -- "$@" --reference-replay "$reference"
  fi
  PYTHONPATH="$PAPER85_REPO/src" "$@"
}

for specification in "2 28" "3 49"; do
  set -- $specification
  session_count=$1
  first_decision=$2
  directory="$PAPER85_OUTPUT/session_${session_count}"
  full="$directory/full_persistent.json"
  run_replay "$session_count" "$first_decision" full "$full"
  run_replay "$session_count" "$first_decision" \
    persistent_episode_retirement "$directory/completed_episode_spine.json" "$full"
  run_replay "$session_count" "$first_decision" \
    persistent_active_episode "$directory/active_episode_only.json" "$full"
done
