#!/usr/bin/env bash
set -euo pipefail

repo=/home/killu/pdattention-paper45-vllm-lifecycle
result_root=/home/killu/pra-cuda-results
cd "$repo"
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
rm -rf "$result_root/paper45-agent-sparse-lifecycle-qwen15-v2"
exec /home/killu/venvs/vllm028/bin/python \
  -m experiments.paper4_5_agent.run_vllm_cuda_live_agent_kv_gate \
  --trajectory task02.traj.json \
  --output "$result_root/paper45-agent-sparse-lifecycle-qwen15-v2.json" \
  --storage "$result_root/paper45-agent-sparse-lifecycle-qwen15-v2" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --turns 10 \
  --continuation-tokens 8 \
  --retention-fraction 0.9 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --detached-reserve-blocks 384
