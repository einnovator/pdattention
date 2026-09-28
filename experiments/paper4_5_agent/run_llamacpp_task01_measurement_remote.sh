#!/bin/zsh
set -eu

# Run one immutable autonomous Task-01 arm against a fresh patched llama.cpp
# process. This launcher is intended for the 48 GB Apple-Silicon host used by
# the Paper 4.5 gate and keeps every arm in a separate output directory.

arm=${1:-pra100}
case "$arm" in
  pra100)
    mode=direct-native-pra
    budget=1.0
    ;;
  pra90)
    mode=direct-native-pra
    budget=0.9
    ;;
  plain)
    mode=no-pra
    budget=1.0
    ;;
  *)
    echo "usage: $0 {plain|pra100|pra90}" >&2
    exit 2
    ;;
esac

runtime=${PRA_AGENT_RUNTIME:-/Users/admin.jorge.simao/git/rd/p45-gate-5fbc}
llama_repo=${PRA_LLAMA_REPO:-/Users/admin.jorge.simao/git/rd/upstream/llama.cpp}
paper67=${PRA_LLAMA_ADAPTER_SRC:-/Users/admin.jorge.simao/git/rd/pdattention-paper6-7-agentkv/src}
python=${PRA_AGENT_PYTHON:-/Users/admin.jorge.simao/.venvs/paper45-agent/bin/python}
model=${PRA_AGENT_MODEL_PATH:-/Users/admin.jorge.simao/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a}
output_root=${PRA_AGENT_OUTPUT_ROOT:-/Users/admin.jorge.simao/git/rd/paper45-easy50-runs/e2e_gate/llamacpp_task01_posttelemetry}
run_token=${PRA_RUN_TOKEN:-p45-llamacpp-task01-posttelemetry-${arm}}
output="$output_root/$arm"
slots="$output_root/slots-$arm"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite immutable output: $output" >&2
  exit 3
fi
mkdir -p "$output_root" "$slots"
export PYTHONPATH="$runtime/src:$runtime:$paper67"

server_pid=
wrapper_pid=
cleanup() {
  [[ -n ${wrapper_pid:-} ]] && kill -TERM "$wrapper_pid" 2>/dev/null || true
  [[ -n ${server_pid:-} ]] && kill -TERM "$server_pid" 2>/dev/null || true
  wait ${wrapper_pid:-999999} 2>/dev/null || true
  wait ${server_pid:-999999} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$llama_repo"
nice -n 5 ./build/bin/llama-server \
  --model "$model" --host 127.0.0.1 --port 18082 -ngl 99 -c 32768 -np 2 \
  --kv-unified --alias qwen3-coder:30b --slot-save-path "$slots" \
  --flash-attn auto --no-webui -t 4 -tb 4 -b 256 -ub 256 \
  > "$output_root/llama-server-$arm.log" 2>&1 &
server_pid=$!
for _ in {1..240}; do
  curl -fsS http://127.0.0.1:18082/health >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS http://127.0.0.1:18082/health >/dev/null

cd "$runtime"
nice -n 5 "$python" -m experiments.paper4_5_agent.serve_llamacpp_pra \
  --llama-url http://127.0.0.1:18082 --host 127.0.0.1 --port 18101 \
  --run-token "$run_token" --mode direct --model qwen3-coder:30b \
  --model-fingerprint "qwen3-coder-30b-q4-k-m-posttelemetry-$arm" \
  --resource-slot 0 --request-slot 1 --prefix-caching --reset-slots \
  --slot-save-path "$slots" --llama-server-pid "$server_pid" \
  > "$output_root/wrapper-$arm.log" 2>&1 &
wrapper_pid=$!
for _ in {1..60}; do
  curl -fsS http://127.0.0.1:18101/health >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS http://127.0.0.1:18101/health >/dev/null

engine_revision=$(git -C "$llama_repo" rev-parse --short=12 HEAD)
"$python" -m experiments.paper4_5_agent.runners.local_qwen_swebench \
  --benchmark-card experiments/paper4_5_agent/benchmarks/swebench_verified_easy50_baseline_success14.json \
  --task-index 1 --output "$output" \
  --run-id "p45-llamacpp-task01-posttelemetry-$arm" \
  --model qwen3-coder:30b --served-model qwen3-coder:30b \
  --model-revision 06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca \
  --tokenizer-revision 06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca \
  --dtype mixed --quantization Q4_K_M --kv-cache-dtype f16 \
  --context-limit 32768 --base-url http://127.0.0.1:18101/v1 \
  --mode "$mode" --budget-fraction "$budget" --sampling-seed 0 --top-p 1.0 \
  --max-completion-tokens 1024 --engine llama.cpp \
  --engine-version "$engine_revision-pra-live-kv-rss" \
  --prefix-caching --require-endpoint-preflight \
  --endpoint-run-token "$run_token" --skip-image-prepull
