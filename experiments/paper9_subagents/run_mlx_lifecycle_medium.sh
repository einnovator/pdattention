#!/bin/sh
set -eu

ROOT=${PAPER9_ROOT:-/private/tmp/pdattention-paper9-lifecycle}
PYTHON=${PAPER9_PYTHON:-/Users/jorge.simao/git/rd/pdattention-crossdoc/.venv/bin/python}
OUTPUT=${PAPER9_OUTPUT:-$ROOT/.runs/mlx_lifecycle_m5_v1}

# Preserve the Paper 3.3 region/layer audit and the Paper 4.5 agent campaign.
# Their long-lived controller PIDs provide an unambiguous hand-off boundary.
while pgrep -f 'experiments.paper3_3_crossdoc_expansion.run_expansion_generation' >/dev/null \
   || pgrep -f 'experiments.paper4_5_agent.runners.local_qwen_swebench' >/dev/null; do
  sleep 60
done

cd "$ROOT"
PYTHONPATH=src:. "$PYTHON" -m experiments.paper9_subagents.run_mlx_lifecycle \
  --model mlx-community/Qwen3-0.6B-4bit \
  --revision 73e3e38d \
  --shared-tokens 2048 \
  --repetitions 5 \
  --output "$OUTPUT"
