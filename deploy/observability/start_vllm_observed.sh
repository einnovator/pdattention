#!/usr/bin/env bash
set -euo pipefail

VLLM_BIN="${VLLM_BIN:-$HOME/venvs/vllm028/bin/vllm}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
OTLP_ENDPOINT="${OTLP_ENDPOINT:-http://192.168.1.102:4317}"
LOG_FILE="${LOG_FILE:-/tmp/pra-vllm-observed.log}"
PID_FILE="${PID_FILE:-/tmp/pra-vllm-observed.pid}"
FOREGROUND="${FOREGROUND:-0}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "vLLM is already running as PID $(cat "$PID_FILE")"
  exit 0
fi

export OTEL_SERVICE_NAME="pra-vllm"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=lab,pra.engine=vllm,pra.model_family=${MODEL},machine.role=rtx-windows"
# vLLM 0.28's V2 runner requires UVA, which WSL does not expose on this host.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
COMMAND=("$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching \
  --otlp-traces-endpoint "$OTLP_ENDPOINT" \
  --collect-detailed-traces model \
  --kv-cache-metrics)

if [[ "$FOREGROUND" == "1" ]]; then
  exec "${COMMAND[@]}"
fi

nohup "${COMMAND[@]}" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "started vLLM PID $(cat "$PID_FILE"); log: $LOG_FILE"
