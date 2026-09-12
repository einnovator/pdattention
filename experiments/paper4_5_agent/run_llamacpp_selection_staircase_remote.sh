#!/bin/zsh
set -eu

# Reproduce the frozen successful Task 01 trajectory through fresh llama.cpp
# processes. Each condition gets an empty engine/session so a prior replay
# cannot influence cache residence or the next condition.

runtime=${PRA_AGENT_RUNTIME:-/Users/admin.jorge.simao/git/rd/pdattention-agent-8c91}
runs=${PRA_AGENT_RUNS:-/Users/admin.jorge.simao/git/rd/paper45-easy50-runs/selection_staircase}
llama_repo=${PRA_LLAMA_REPO:-/Users/admin.jorge.simao/git/rd/upstream/llama.cpp}
python=${PRA_AGENT_PYTHON:-/Users/admin.jorge.simao/.venvs/paper45-agent/bin/python}
paper67=${PRA_LLAMA_ADAPTER_SRC:-/Users/admin.jorge.simao/git/rd/pdattention-paper6-7-agentkv/src}
model=${PRA_AGENT_MODEL_PATH:-/Users/admin.jorge.simao/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a}
interaction_history=${PRA_AGENT_HISTORY:-/Users/admin.jorge.simao/git/rd/paper45-easy50-runs/e2e_gate/llamacpp_task01_pra100_v1/interaction_history.jsonl}
turns=${PRA_AGENT_REPLAY_TURNS:-31}
condition_list=${PRA_AGENT_STAIRCASE_CONDITIONS:-"0 1"}
forced_bundle_start=${PRA_AGENT_FORCED_BUNDLE_START:-}

mkdir -p "$runs"
mkdir -p "$runs/slots"
export PYTHONPATH="$runtime/src:$runtime:$paper67"

server_pid=
wrapper_pid=
cleanup() {
  [[ -n ${wrapper_pid:-} ]] && kill -TERM "$wrapper_pid" 2>/dev/null || true
  [[ -n ${server_pid:-} ]] && kill -TERM "$server_pid" 2>/dev/null || true
  wait ${wrapper_pid:-999999} 2>/dev/null || true
  wait ${server_pid:-999999} 2>/dev/null || true
  server_pid=
  wrapper_pid=
}
trap cleanup EXIT INT TERM

start_engine() {
  local label=$1
  cd "$llama_repo"
  nice -n 5 ./build/bin/llama-server \
    --model "$model" --host 127.0.0.1 --port 18082 -ngl 99 -c 32768 -np 2 \
    --kv-unified --alias qwen3-coder:30b --slot-save-path "$runs/slots" \
    --flash-attn auto --no-webui -t 4 -tb 4 -b 256 -ub 256 \
    > "$runs/llama-server-$label.log" 2>&1 &
  server_pid=$!
  local ready=0
  for _ in {1..240}; do
    if curl -fsS http://127.0.0.1:18082/health >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ $ready == 1 ]] || return 3

  cd "$runtime"
  nice -n 5 "$python" -m experiments.paper4_5_agent.serve_llamacpp_pra \
    --llama-url http://127.0.0.1:18082 --host 127.0.0.1 --port 18101 \
    --mode direct --model qwen3-coder:30b \
    --model-fingerprint "qwen3-coder-30b-q4-k-m-$label" \
    --resource-slot 0 --request-slot 1 --prefix-caching --reset-slots \
    > "$runs/wrapper-$label.log" 2>&1 &
  wrapper_pid=$!
  ready=0
  for _ in {1..60}; do
    if curl -fsS http://127.0.0.1:18101/health >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ $ready == 1 ]] || return 4
}

run_condition() {
  local omitted=$1
  local label="omit-${omitted}-bundles"
  local forced_args=()
  if [[ -n $forced_bundle_start ]]; then
    label="forced-bundle-${forced_bundle_start}"
    forced_args=(--forced-bundle-start "$forced_bundle_start")
  fi
  start_engine "$label"
  cd "$runtime"
  set +e
  "$python" -m experiments.paper4_5_agent.run_selection_staircase_replay \
    --interaction-history "$interaction_history" \
    --output "$runs/$label.json" \
    --base-url http://127.0.0.1:18101/v1 \
    --model qwen3-coder:30b --engine llama.cpp \
    --max-omitted-bundles "$omitted" ${forced_args[@]} --turns "$turns"
  local run_status=$?
  set -e
  cleanup
  return $run_status
}

for omitted in ${=condition_list}; do
  run_condition "$omitted"
done
