#!/bin/sh
set -eu

prompt_file="${P85_TASK_PROMPT_FILE:-/paper85/task_prompt.txt}"
workdir="${P85_WORKDIR:-/testbed}"
model="${P85_MODEL:-openai-compatible/qwen3-coder:30b}"

if [ ! -s "$prompt_file" ]; then
    echo "missing task prompt: $prompt_file" >&2
    exit 2
fi

prompt=$(cat "$prompt_file")
exec kilo --pure run \
    --format json \
    --auto \
    --model "$model" \
    --dir "$workdir" \
    "$prompt"
