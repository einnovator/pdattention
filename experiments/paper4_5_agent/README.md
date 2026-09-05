# Paper 4.5 agent benchmark

This campaign enforces the experiment order in the Paper 4.5 benchmark
contract: published result, ordinary no-PRA reproduction, official grading,
compatibility review, gateway pass-through, and only then PRA conditions.

The existing `experiments.agents` package remains the shared normalized result
and official-harness adapter layer. This package adds long-running campaign
state, external-result provenance, admission, and crash-safe reports.

## Easier capability ladder

Paper 4.5 first calibrates the available local model on deterministic subsets
of the official SWE-bench Verified `<15 min fix` difficulty stratum. The
Easy-20 and Easy-50 cards are nested, revision-pinned, digest-protected, and
created before model execution:

```bash
python -m experiments.paper4_5_agent.build_easy_cohorts
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/swebench_easy20_calibration.yaml \
  --max-hours 18 --resume
```

The initial cell is an official No-PRA local calibration. It is not a claimed
reproduction of the published H100 result. A score below 20% or above 80%
closes the treatment gate; 30%-70% is the preferred operating regime. Only a
qualifying frozen baseline permits an Easy-50 pass-through, matched-truncation,
or PRA Selected Context comparison.

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

The completed v3 campaign produced 30/30 official trials: Mini-SWE-agent and
Qwen Code each solved one task, Aider solved none, and only `fix-git` was solved
by any harness. Aggregate success was 2/30 (6.7%), below the configured 10%
promotion floor, so no PRA treatment was launched. The 20 runs whose harnesses
reported token usage accumulated 6,717,096 input tokens. Normalized evidence
and the complete compressed Harbor artifact tree are under
`docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/`
`qwen3_coder_30b_multi_harness_v3/`.

## Controlled SWE-bench fixed-50 campaign

The next campaign uses the exact ordered 50-instance cohort from the external
controlled local-model study. Its machine-readable card, source commit, and
platform-independent digest are in
`benchmarks/swebench_verified_fixed50.json`. The source recipe is encoded in
the model, engine, harness, and PRA YAML files under `configs/`.

Run a cheap host audit before installing or loading any model:

```bash
python -m experiments.paper4_5_agent.runners.swebench_verified \
  --benchmark-card experiments/paper4_5_agent/benchmarks/swebench_verified_fixed50.json \
  --output /tmp/pra-swebench-preflight \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model Qwen3-Coder-30B-A3B \
  --model-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --tokenizer-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --run-id qwen3-preflight --preflight-only
```

The source-matched run requires mini-swe-agent `2.4.0`, SWE-bench `4.1.0`,
vLLM `0.22.1`, and one H100 80 GB. The source did not publish immutable model
or tokenizer revisions, so these source limitations remain explicit in every
preflight receipt. The execution configurations pin current immutable HF
revisions and installed package `RECORD` digests. `--allow-partial-reproduction`
permits diagnostics on a changed host, but any actual configuration difference
prevents that receipt from unlocking PRA.
The runner also verifies the pinned SWE-bench dataset revision and uses a
campaign-local datasets cache so an older shared cache cannot silently satisfy
the run.

The full resumable scheduler is intentionally disabled until a qualifying host
is available:

```bash
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/swebench_pra_frontier.yaml \
  --max-hours 16 --resume
```

Once enabled, it runs Qwen first, then Gemma, and admits the 50/25/12.5 percent
frontier only when Gemma is officially reproduced and its observed baseline
score is at least 20 percent. The Qwen 14 percent target remains a lower-
capability control and cannot unlock the Gemma treatment cells.

The treatment runner no longer contains placeholder commands. Start the pinned
vLLM model on port `8000`, then expose two separately identified gateway modes:

```bash
pra gateway serve --mode passthrough --backend vllm \
  --backend-url http://127.0.0.1:8000/v1 --model google/gemma-4-31B-it \
  --pra-bundle none --port 8080
pra gateway serve --mode selected-context --backend vllm \
  --backend-url http://127.0.0.1:8000/v1 --model google/gemma-4-31B-it \
  --pra-bundle none --port 8081
```

For every mini-swe-agent request, the treatment proxy keeps all system messages
and the current user observation mandatory. Matched truncation fills the
remaining budget from recent history. Selected Context segments only earlier
agent-visible messages, ranks them with the runtime's typed/BM25/hashed-
embedding RRF index, and uses recency only to fill an unused budget. It sends
the selected segments as typed resources to G10 and records a
`request_telemetry.jsonl` row with logical, selected, visible, avoided-token,
and routing-time estimates. These estimates use a declared whitespace counter;
engine-reported tokenizer and timing metrics remain separate when available.

After official grading, the runner joins each task to its mini-swe-agent
`*.traj.json` by instance ID. It records exact backend prompt/completion usage,
model and tool calls, maximum prompt size, conservative repeated-context load,
trajectory duration, termination, and patch size. A stable hash of the first
agent-visible task message joins treatment requests to the task row. Endpoint
timings such as TTFT and prefill remain null unless the serving layer reports
them; the normalizer does not infer those measurements from wall time.
