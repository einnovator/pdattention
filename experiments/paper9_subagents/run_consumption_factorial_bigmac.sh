#!/bin/sh
set -eu

ROOT=${PAPER9_ROOT:-/private/tmp/pdattention-paper9-consumption}
PYTHON=${PAPER9_PYTHON:-/Users/admin.jorge.simao/git/rd/pdattention-runtime-night/.venv/bin/python}
OUTPUT=${PAPER9_OUTPUT:-$ROOT/.runs/consumption_factorial_v1}
OLLAMA_BIN=${OLLAMA_BIN:-/Applications/Ollama.app/Contents/Resources/ollama}
OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11435}
export OLLAMA_HOST

# Paper 4.5 owns the loaded 30B model. Its 30-minute keep-alive expiry is the
# conservative hand-off signal: do not compete with active coding-agent work.
while "$OLLAMA_BIN" ps | grep -q 'qwen3-coder:30b'; do
  sleep 60
done

cd "$ROOT"
PYTHONPATH=src:. "$PYTHON" -m experiments.paper9_subagents.run_consumption_factorial \
  --manifest experiments/paper9_subagents/benchmarks/consumption_factorial_v1.json \
  --output "$OUTPUT" \
  --base-url http://127.0.0.1:11435 \
  --models qwen3-coder:30b,qwen3:14b,gemma3:4b-it-qat \
  --max-steps 6 \
  --max-new-tokens 384
