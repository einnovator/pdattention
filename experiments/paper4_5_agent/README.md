# Paper 4.5 agent benchmark

This campaign enforces the experiment order in the Paper 4.5 benchmark
contract: published result, ordinary no-PRA reproduction, official grading,
compatibility review, gateway pass-through, and only then PRA conditions.

The existing `experiments.agents` package remains the shared normalized result
and official-harness adapter layer. This package adds long-running campaign
state, external-result provenance, admission, and crash-safe reports.

```bash
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/fim14b_r2egym.yaml \
  --max-hours 16 --resume
```

Use `--dry-run` to validate dependencies and create the report skeleton. A
non-baseline cell is marked `BLOCKED` until its exact no-PRA dependency is
`BASELINE_REPRODUCED`. A smoke subset, changed quantization, changed engine, or
unofficial grader can be useful diagnostics, but remains
`BASELINE_ATTEMPTED`.

Distributed hosts can return their normalized `official_result.json` without
replaying inference locally:

```bash
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/fim7b_q4_calibration.yaml \
  --record-cell fim7b-q4-no-pra-smoke \
  --record-result path/to/official_result.json
```

The two-task default is only a path smoke. Expand the unchanged No-PRA cohort
before scheduling any gateway or PRA condition:

```bash
COUNT=20 START_IDX=0 RUN_ID=fim7b-q4-no-pra-baseline-20 \
OUTPUT=$HOME/experiments/paper4_5_agent/fim7b_q4_no_pra_baseline_20 \
bash experiments/paper4_5_agent/runners/run_fim7b_q4_smoke.sh
```

Per-task rows retain cumulative/max prompt tokens, conservative unique and
repeated-context estimates, model/tool calls, trajectory length, patch size,
model/tool/wall time, termination, and official grader outcome. Engine TTFT
and prefill are `null` unless measured directly. A larger altered-engine cohort
still remains `BASELINE_ATTEMPTED`; only the exact published cell can unlock
treatments.

## Cross-harness qualification

The stronger-model pilot runs the same 10 frozen Terminal-Bench 2.1 tasks with
Qwen3-Coder-30B through Mini-SWE-agent, Qwen Code, and Aider. It is
deliberately No-PRA:
the matrix must complete all 30 official Harbor trials before its admission
gate can pass. Each trial has its own Harbor directory and normalized row, so a
host restart resumes at the first incomplete `(model, task, harness)` cell.

The portable cohort excludes browser, video, and Windows-specific verifiers.
During host qualification, the amd64 Selenium verifier for
`filter-js-from-html` reached 92% of the ARM Docker host's memory limit and
roughly 1,900 emulated processes without completing. Its receipt is retained as
a host limitation and is not counted as a model outcome.

OpenCode and Pi remain represented by the earlier WSL-hosted mechanism gates.
Their Node processes did not terminate after final output under Rosetta, so
those harnesses are not silently mixed into this ARM-host cohort.
SWE-agent 1.1.0 and OpenHands 0.57.0 also failed host qualification before a
usable model trajectory: Harbor passed SWE-agent a literal `$(pwd)` repository
path, while OpenHands exhausted its API retry loop. The matrix marks such
zero-call failures `INVALID`, keeps them retryable, and excludes them from
quality and admission statistics.

```bash
export PRA_AGENT_QWEN3_CODER_30B_URL=http://MODEL_HOST:11435/v1
python -m experiments.paper4_5_agent.harness_matrix \
  --config experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pilot.yaml \
  --resume
```

Use `--max-cells 15` for the first five-task cross-harness checkpoint. The
matrix stores no API credential, marks `pra_enabled=false` in every receipt,
and reports task success separately by agent. This comparison qualifies the
harness/model combinations; it is not an agent leaderboard because prompts,
tool schemas, and loop policies differ.
