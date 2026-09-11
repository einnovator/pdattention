#!/usr/bin/env bash
set -euo pipefail

repo=/home/killu/pdattention-paper45-vllm-rtx
result_root=/home/killu/pra-cuda-results
cd "$repo"
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
rm -rf "$result_root/paper45-agent-sparse-v1"
exec /home/killu/venvs/vllm028/bin/python \
  -m experiments.paper4_5_agent.run_vllm_cuda_live_agent_kv_gate \
  --trajectory task02.traj.json \
  --output "$result_root/paper45-agent-sparse-v1.json" \
  --storage "$result_root/paper45-agent-sparse-v1" \
  --turns 4 \
  --continuation-tokens 8 \
  --retention-fraction 0.9 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.40 \
  --detached-reserve-blocks 384
