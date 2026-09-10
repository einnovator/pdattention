#!/bin/sh
set -eu

ROOT=${PAPER4_5_ROOT:-/private/tmp/pdattention-paper4-agent-eval}
PYTHON=${PAPER4_5_PYTHON:-/private/tmp/paper45-agent-venv/bin/python}
OUTPUT=${PAPER4_5_OUTPUT:-$ROOT/.runs/easy50_gateway_passthrough_corrected}
PAPER9_SENTINEL_URL=${PAPER9_SENTINEL_URL:-http://192.168.1.6:8765/mlx_lifecycle_m4_v1/summary.json}
GATEWAY_URL=${PRA_AGENT_G00_URL:-http://192.168.1.6:8180}
PATH=/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:/opt/homebrew/bin:$PATH
export PATH
PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS=1
export PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS

# Preserve Paper 9's factorial and direct-MLX lifecycle measurements on the M4.
until curl -fsS "$PAPER9_SENTINEL_URL" >/dev/null 2>&1; do
  sleep 60
done

# Docker can be restarted while Paper 9 runs, but do not resume the official
# harness until its daemon is actually ready.
until docker info >/dev/null 2>&1; do
  sleep 60
done

cd "$ROOT"
PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m experiments.paper4_5_agent.runners.local_qwen_swebench \
  --benchmark-card experiments/paper4_5_agent/benchmarks/swebench_verified_easy50.json \
  --output "$OUTPUT" \
  --run-id easy50-gateway-passthrough-corrected \
  --mode gateway-passthrough \
  --base-url "$GATEWAY_URL"
