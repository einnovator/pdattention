#!/bin/sh
set -eu

ROOT=${PAPER4_5_ROOT:-/private/tmp/pdattention-paper4-agent-eval}
PYTHON=${PAPER4_5_PYTHON:-/private/tmp/paper45-agent-venv/bin/python}
GATEWAY_OUTPUT=${PAPER4_5_GATEWAY_OUTPUT:-$ROOT/.runs/easy50_gateway_passthrough_corrected}
OUTPUT=${PAPER4_5_TRUNCATION_OUTPUT:-$ROOT/.runs/easy50_truncation_50_matched}
DIRECT_URL=${PRA_AGENT_DIRECT_URL:-http://192.168.1.6:11435/v1}
PATH=/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:/opt/homebrew/bin:$PATH
export PATH
PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS=1
export PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS

# Preserve the predeclared ordering: qualify G00 on all 50 tasks before the
# matched 50-percent truncation arm begins.
until test -s "$GATEWAY_OUTPUT/official_result.json"; do
  sleep 60
done

# Bind the direct arm to the same frozen Ollama model identity used by the
# admitted baseline and corrected G00 gateway.
"$PYTHON" - "$GATEWAY_OUTPUT/official_result.json" "$DIRECT_URL" <<'PY'
import json
import sys
import urllib.request

gateway_result = json.load(open(sys.argv[1], encoding="utf-8"))
if not gateway_result.get("official_grader") or gateway_result.get("total") != 50:
    raise SystemExit("Corrected G00 result is not a complete official Easy50 receipt.")

root = sys.argv[2].rstrip("/").removesuffix("/v1")
with urllib.request.urlopen(root + "/api/version", timeout=15) as response:
    version = json.load(response).get("version")
if version != "0.32.7":
    raise SystemExit(f"Unexpected Ollama version: {version!r}")

with urllib.request.urlopen(root + "/api/tags", timeout=15) as response:
    models = json.load(response).get("models", [])
match = next((row for row in models if row.get("name") == "qwen3-coder:30b"), None)
expected = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"
if match is None or match.get("digest") != expected:
    raise SystemExit(f"Direct model identity mismatch: {match!r}")
PY

until docker info >/dev/null 2>&1; do
  sleep 60
done

cd "$ROOT"
PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" \
  -m experiments.paper4_5_agent.runners.local_qwen_swebench \
  --benchmark-card experiments/paper4_5_agent/benchmarks/swebench_verified_easy50.json \
  --output "$OUTPUT" \
  --run-id easy50-truncation-50-matched \
  --mode truncation \
  --budget-fraction 0.5 \
  --base-url "$DIRECT_URL"
