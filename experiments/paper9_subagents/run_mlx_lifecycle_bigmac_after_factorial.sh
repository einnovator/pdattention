#!/bin/sh
set -eu

ROOT=${PAPER9_ROOT:-/private/tmp/pdattention-paper9-consumption}
PYTHON=${PAPER9_PYTHON:-/Users/admin.jorge.simao/git/rd/pdattention-runtime-night/.venv/bin/python}
FACTORIAL_OUTPUT=${PAPER9_FACTORIAL_OUTPUT:-$ROOT/.runs/consumption_factorial_v1}
OUTPUT=${PAPER9_OUTPUT:-$ROOT/.runs/mlx_lifecycle_m4_v1}

# Run only after the Big Mac consumption factorial has completed successfully.
# If its controller disappears without a summary, fail instead of silently
# producing lifecycle evidence after an incomplete prerequisite campaign.
while [ ! -f "$FACTORIAL_OUTPUT/summary.json" ]; do
  if ! pgrep -f 'run_consumption_factorial_bigmac.sh|run_consumption_factorial' >/dev/null; then
    echo "Consumption factorial stopped without producing $FACTORIAL_OUTPUT/summary.json" >&2
    exit 2
  fi
  sleep 60
done

cd "$ROOT"
PYTHONPATH=src:. "$PYTHON" -m experiments.paper9_subagents.run_mlx_lifecycle \
  --model mlx-community/Qwen3-0.6B-4bit \
  --revision 73e3e38d \
  --shared-tokens 2048 \
  --repetitions 5 \
  --output "$OUTPUT"
