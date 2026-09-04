#!/usr/bin/env bash
set -euo pipefail

R2E_ROOT="${R2E_ROOT:-$HOME/git/rd/R2E-Gym}"
OUTPUT="${OUTPUT:-$HOME/experiments/paper4_5_agent/fim7b_q4_no_pra_smoke}"
RUNNER="${RUNNER:-/mnt/c/Users/killu/paper4_5_r2egym.py}"

mkdir -p "$OUTPUT"
exec "$R2E_ROOT/.venv/bin/python" "$RUNNER" \
  --r2egym-root "$R2E_ROOT" \
  --output "$OUTPUT" \
  --base-url http://192.168.1.102:8400/v1 \
  --model TIGER-Lab/FIM-7B \
  --model-revision 5a1d4294185e4fa0bbd40750c87d0beab7e67a3a \
  --served-model FIM-7B \
  --engine llama.cpp \
  --engine-version docker-inference-build-recorded-in-run-manifest \
  --quantization Q4_K_M \
  --harness-version 0d94c4eb9431cd195c55a7ea3abd54006c9a1735 \
  --run-id fim7b-q4-no-pra-smoke \
  --count 2 \
  --workers 1 \
  --context-limit 32768 \
  --max-steps 40 \
  --max-steps-absolute 100 \
  --grader-timeout 1200 \
  --configuration-difference engine=llama.cpp-not-vLLM \
  --configuration-difference quantization=Q4_K_M-not-BF16 \
  --configuration-difference cohort=2-not-500 \
  --configuration-difference context=32768-not-65536
