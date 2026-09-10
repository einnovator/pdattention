#!/bin/sh
set -eu

ROOT=${PAPER4_5_ROOT:-/private/tmp/pdattention-paper4-agent-eval}
PYTHON=${PAPER4_5_PYTHON:-/private/tmp/paper45-agent-venv/bin/python}
TRUNCATION_OUTPUT=${PAPER4_5_TRUNCATION_OUTPUT:-$ROOT/.runs/easy50_truncation_50_matched}
OUTPUT=${PAPER4_5_G10_OUTPUT:-$ROOT/.runs/easy50_g10_selected_50_matched}
G10_URL=${PRA_AGENT_G10_URL:-http://192.168.1.6:8182}
PATH=/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:/opt/homebrew/bin:$PATH
export PATH
PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS=1
export PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS

# Preserve the matched efficacy sequence: the 50-percent truncation control
# must finish before selected context is evaluated at the same physical budget.
until test -s "$TRUNCATION_OUTPUT/official_result.json"; do
  sleep 60
done

"$PYTHON" - "$TRUNCATION_OUTPUT/official_result.json" "$G10_URL" <<'PY'
import json
import sys
import urllib.request

truncation = json.load(open(sys.argv[1], encoding="utf-8"))
if not truncation.get("official_grader") or truncation.get("total") != 50:
    raise SystemExit("Matched truncation result is not a complete official Easy50 receipt.")

root = sys.argv[2].rstrip("/").removesuffix("/v1")
with urllib.request.urlopen(root + "/health", timeout=15) as response:
    health = json.load(response)
effective = health.get("effective_capabilities") or {}
if health.get("status") != "ok" or health.get("gateway_mode") != "G10":
    raise SystemExit(f"G10 endpoint identity mismatch: {health!r}")
if not effective.get("logical_refs") or not effective.get("typed_records"):
    raise SystemExit(f"G10 selected-context capabilities missing: {effective!r}")
PY

until docker info >/dev/null 2>&1; do
  sleep 60
done

cd "$ROOT"
PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" \
  -m experiments.paper4_5_agent.runners.local_qwen_swebench \
  --benchmark-card experiments/paper4_5_agent/benchmarks/swebench_verified_easy50.json \
  --output "$OUTPUT" \
  --run-id easy50-g10-selected-50-matched \
  --mode gateway-pra \
  --budget-fraction 0.5 \
  --base-url "$G10_URL"
