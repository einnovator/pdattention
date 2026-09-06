# Agent Benchmarks

The coding-agent campaign asks whether PRA lowers cumulative context-processing
cost while preserving official task success. End-task success is primary;
tokens, latency, resources, and cost are interpreted only beside that result.
External code or documentation RAG is outside this campaign.

## Staged campaign

1. **Stage A, compatibility:** 2-5 trivial tasks verify unattended execution,
   tool use, protocol behavior, isolation, and accounting. No quality claims.
2. **Stage B, baseline admission:** OpenCode, Pi, and OpenHands first run 10-25
   frozen tasks under No PRA. Only agent/model/task cohorts with roughly
   30%-80% official success proceed to the PRA conditions.
3. **Stage C, profiles:** stable pairs expand to QUALITY, BALANCED, and ECONOMY
   only after the initial No PRA, PRA - No Adaptor/BALANCED, and PRA - Adaptor
   Bundle/BALANCED comparison is useful.
4. **Stage D/E, breadth:** commercial-native controls and larger official
   benchmark cohorts are added only after the pilot is stable.

## Reproduce before enabling PRA

The long-running campaign is controlled by
`experiments.paper4_5_agent.run_campaign`. Each cell pins the published score,
source revision, benchmark cohort, model and tokenizer revisions, harness,
engine, precision, scaffold, context limit, decoding policy, step limit, and
official grader. The scheduler writes durable state and four partial reports
after every transition, so an interrupted run can resume without repeating
completed instances.

```bash
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/fim14b_r2egym.yaml \
  --max-hours 16 --resume
```

`gateway_passthrough`, `gateway_pra`, and `native_pra` cells must name an exact
no-PRA dependency. They are blocked unless that dependency has been officially
graded and marked `BASELINE_REPRODUCED`. A changed engine, quantization,
context limit, task cohort, or unofficial grader remains
`BASELINE_ATTEMPTED`, even when it is operationally useful.

## Separate gateway placement from PRA execution

The agent comparison uses two independent switches: whether a PRA gateway is
present and whether the engine performs native PRA. This produces four controls
before budget/profile variants are considered:

| Condition | Connection | Gateway PRA | Engine PRA | Interpretation |
| --- | --- | --- | --- | --- |
| Direct ordinary | agent -> engine | off | off | No-PRA quality and cost baseline. |
| G00 pass-through | agent -> gateway -> engine | off | off | Gateway transport overhead only. |
| Direct native PRA | agent -> PRA engine | off | on | Native PRA without gateway mediation. |
| G11 native PRA | agent -> PRA gateway -> PRA engine | on | on | Full gateway plus native-engine product path. |

G10 Selected Context is retained as a useful text-fallback condition, but it is
not the full gateway-plus-native-engine arm. The principal gateway comparison
is G11 minus direct native PRA, using the same engine instance and matched task,
repeat, model revision, selected records, budget, and profile. A version-2 run
records `connection`, `engine_pra_enabled`, `gateway_pra_enabled`,
`gateway_mode`, `engine_target_id`, and `comparison_group` in every normalized
row. Capability preflight rejects Native Memory when `native_kv` is not
effective.

The rollout order is Mini-SWE-agent and Qwen Code first, followed by admitted
OpenCode, Pi, OpenHands, Aider, and SWE-agent configurations. Commercial agents
are included only when their endpoint is genuinely configurable; Codex uses the
Responses gateway path, while Claude Code and Gemini CLI remain blocked under
their currently audited protocols. PRA Agent is the final first-party arm and
runs twice, direct and through G11. Besides comparison, disagreement on that
pair is a diagnostic for PRA Agent session, tool, typed-record, or gateway
behavior rather than an engine-quality effect.

Gateway treatments have an additional transport gate. Before launching an
agent, the runner requires a healthy endpoint, the expected G00 or G10 mode,
and the exact served model in `/v1/models`. G00 preserves OpenAI generation
options and the upstream completion envelope. Upstream HTTP and connection
failures are returned as structured JSON errors instead of terminating the
client connection. A failed transport gate is an invalid treatment run, never
a zero-quality PRA result.

The first Easy-50 G00 attempt is retained as such an invalid run: all `50/50`
trajectories stopped on the first call with a transport error and no
submission. After repairing URL normalization and pass-through fidelity, a
clean three-task gate reran identities previously solved directly and again
resolved `3/3` with the official grader. The corrected gate used 69 model
calls, 403,703 prompt tokens, and 9,503 output tokens, with no timeouts or
grader errors. This selected-on-success smoke establishes end-task transport
parity; it is not an estimate of gateway accuracy or latency. A fresh Easy-50
G00 run remains the population-level comparison.

The primary target is
[`TIGER-Lab/FIM-14B`](https://huggingface.co/TIGER-Lab/FIM-14B), whose model
card reports `29.20%` on all 500 SWE-bench Verified instances with the
R2E-Gym scaffold. The exact BF16/vLLM cell does not fit the available 8 GiB
CUDA host. A no-PRA FIM-7B Q4_K_M/llama.cpp smoke on the 48 GiB M4 therefore
qualifies the distributed model, agent, Docker, and grader path only; its
configuration differences prevent it from unlocking PRA experiments.

The FIM-7B calibration resolved one of two official SWE-bench Verified tasks
with no grader errors. One task terminated naturally after 31 calls and the
other reached the 40-step limit. Across both tasks the unmodified R2E-Gym
agent made 72 model calls, accumulated 1,123,772 prompt tokens and 11,878
output tokens, and recorded 607.4 seconds of trajectory time. The observed
`1/2` has a wide Wilson 95% interval (`9.5-90.5%`) and is not compared as an
accuracy estimate with the published 500-task `17.8%` result. Its campaign
status is `BASELINE_ATTEMPTED`; no gateway or PRA treatment was run.

The next run expands the same No-PRA cell before changing transport. A baseline
cohort must be large enough to show both a nontrivial success rate and stable
trajectory distributions; `1/2` cannot unlock PRA. The runner now preserves
cumulative and maximum prompt tokens, a conservative unique-context estimate,
repeated-context fraction, model and tool calls, trajectory length, patch size,
model/tool/wall time, termination, and official grader outcome. Endpoint TTFT
and prefill stay `NOT_MEASURED` unless the engine exports them. The current
two trajectories contain `50,106` maximum-prompt tokens across tasks versus
`1,123,772` cumulative prompt tokens. Under R2E-Gym's accumulating trajectory
semantics, this is a conservative `95.5%` repeated-context estimate and a
primary later treatment target. Token reduction is meaningful only at a
matched solved-task rate.

Run the expanded baseline with the same frozen harness settings:

```bash
COUNT=20 START_IDX=0 RUN_ID=fim7b-q4-no-pra-baseline-20 \
OUTPUT=$HOME/experiments/paper4_5_agent/fim7b_q4_no_pra_baseline_20 \
bash experiments/paper4_5_agent/runners/run_fim7b_q4_smoke.sh
```

Raw trajectories, predictions, the official grader report, the normalized
receipt, per-task telemetry, and the run manifest are retained under
`docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/fim7b_q4_calibration/`.

## Find a measurable no-PRA task band

The present official runs do not yet admit a PRA efficacy comparison:
Qwen3-14B/OpenCode scored `0/5`, Qwen3-Coder-30B/OpenCode scored `0/1`, and the
deterministic Stage-A fixture is deliberately too easy. The next baseline
screen therefore targets a `30%-80%` no-PRA success band rather than choosing
tasks only because they are available.

`PRA-Coding-Tasks-v1` is the recommended controlled screening set: 24-40
small repository tasks spanning unit-test repair, one-function implementation,
parser fixes, CLI options, configuration precedence, small multi-file
refactors, type-check fixes, API call-site updates, validation, and executable
documentation examples. Every task must freeze its repository revision, test
command, timeout, allowed scope, and checksum. This set is for screening and
mechanism attribution; SWE-bench Lite/Verified, Commit0, RepoQA, and
Terminal-Bench remain the external validation layers.

Once a baseline enters the admission band, the first comparison is only No
PRA, PRA - No Adaptor/BALANCED, and PRA - Adaptor Bundle/BALANCED. The report
must show absolute and baseline-relative task success, cumulative tokens,
TTFT p50/p95/p99, ITL p50/p95/p99, decode output tokens/s, model requests/s,
queue/inference time, task wall time, peak memory, and cost. Missing engine
telemetry remains `NOT_MEASURED`.

The same condition grammar used by model cards is visible in the
[Canonical Evidence Matrix](../bundles/evidence-matrix.md).

Terminal-Bench 2.1 is primary. SWE-bench Lite is the development secondary and
SWE-bench Verified is reserved for promoted configurations. Their official
harnesses remain responsible for task setup and grading.

## Current Stage A evidence

The 2026-09-02 gateway cohort used the same two exact-output tasks twice for
each agent. Ollama/Qwen3-14B ran on an M5 Mac; the isolated agent workspaces and
gateway ran on Windows.

| Agent | Protocol | Runs | Success | Input tokens | Mean task time |
| --- | --- | ---: | ---: | ---: | ---: |
| Codex CLI 0.147.0 | Responses | 4 | 4/4 | 97,958 | 47.3 s |
| OpenCode 1.18.26 | Chat Completions | 4 | 4/4 | 53,170 | 107.7 s |
| Pi 0.73.1 | Chat Completions | 4 | 4/4 | 14,482 | 44.0 s |

These token totals include different agent system prompts and tool schemas, so
they are not a same-agent efficiency comparison. The fixture is intentionally
too small for latency conclusions. Raw and normalized evidence lives under
`docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/`.

## First official task cohort

Harbor 0.22.0 first qualified the Terminal-Bench 2.1 execution path with a
`1/1` oracle result on `filter-js-from-html`. OpenCode 1.18.26 and
Ollama/Qwen3-14B then completed the five frozen smoke tasks through the gateway
with no harness exceptions. Official task success was `0/5`, despite passing
`7/12` verifier tests and using 100,883 cumulative input tokens.

This result does not compare PRA modes. It rejects this general-model pairing
at the Stage-B promotion gate: a code-specialized model must first establish a
nonzero quality floor before context profiles are swept. Keeping quality ahead
of token savings prevents an efficient but ineffective agent from looking
competitive.

The canonical evidence record makes the gate machine-readable. Missing
treatment cells are not zero-valued failures: they are `BLOCKED` because the
baseline cannot support a causal PRA comparison.

| Cohort | Metric | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenCode / Qwen3-14B / five tasks | Official task success | 0/5 | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| OpenCode / Qwen3-14B / five tasks | Verifier checks | 7/12 | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| OpenCode / Qwen3-14B / five tasks | Cumulative input tokens | 100,883 | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| OpenCode / Qwen3-Coder-30B / one task | Official task success | 0/1 | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` |
| OpenCode / Qwen3-Coder-30B / one task | Verifier checks | 4/6 | `BLOCKED` | `BLOCKED` | `BLOCKED` | `BLOCKED` |

The machine-readable records are
`qwen3_14b_canonical_evidence.json` and
`qwen3_coder_30b_canonical_evidence.json` in the campaign artifact directory.
They preserve metric units and direction, exact run IDs, the admission reason,
and explicit missing states.

Qwen2.5-Coder-7B was also checked on the two closest-to-success tasks. It
scored `0/2` and exposed a different incompatibility: intended tool calls were
returned as plain text, leaving the workspace unchanged. The campaign therefore
moves to a same-model, different-agent-loop gate before selecting PRA profiles.

Qwen3-Coder-30B was then tested with OpenCode on `query-optimize`. It used the
tool path correctly but still scored `0/1` (`4/6` verifier tests). Its final SQL
retained the expensive correlated subquery, so the official verifier consumed
almost its full 30-minute allowance. The run used 50,876 cumulative input tokens
and completed without a harness exception. This rules out gateway transport as
the immediate problem, but it still does not provide the nonzero success floor
required for a PRA profile comparison.

That gate used the same `query-optimize` task, Qwen3-14B model, Ollama backend,
gateway, and official verifier for three harnesses:

| Agent | Official success | Verifier tests | Input tokens | Observed behavior |
|---|---:|---:|---:|---|
| OpenCode 1.18.26 | 0/1 | 5/6 | 20,996 | Read and wrote a candidate query |
| Pi 0.73.1 | 0/1 | 4/6 | 5,219 | Three model calls; one read and one write |
| OpenHands 0.57.0 | 0/1 | 2/6 | `NOT_REPORTED` | Repeated continuation without creating the solution |

This is a one-task mechanism check, not a quality ranking. It demonstrates
that harness policy can change token use and partial verifier progress even
when model, endpoint, and task are held fixed. No row establishes a sufficient
quality floor for the PRA profile comparison yet.

## Reproduce qualification

Audit installed agents and validate the deterministic fixture first:

```bash
python -m experiments.agents audit --output artifacts/agent-audit.json
python -m experiments.agents run \
  --manifest fixture_smoke.yaml \
  --output artifacts/fixture
```

Run an external agent only after pinning its executable and provider. The
runner supports repeated `--agent-arg` values for provider-specific flags and
never copies `--env` values into normalized artifacts.

Generate official-harness plans without pretending they have executed:

```bash
python -m experiments.agents plan \
  --manifest terminal_bench_pilot.yaml \
  --agent opencode --model MODEL --condition selected-balanced

python -m experiments.agents plan \
  --manifest swebench_lite_smoke.yaml \
  --agent pi --model MODEL --condition no-pra
```

Normalize completed Harbor jobs while retaining Harbor's official score:

```bash
python -m experiments.agents import-harbor HARBOR_JOB_DIR \
  --manifest terminal_bench_smoke.yaml --output HARBOR_JOB_DIR \
  --engine ollama --engine-version VERSION --host AGENT_AND_ENGINE_HOSTS \
  --hardware engine_chip='"CHIP"' --hardware engine_memory_gib=MEMORY \
  --model MODEL --quantization QUANTIZATION \
  --connection gateway --protocol openai-chat-completions
```

Apply the admission gate before scheduling either PRA treatment:

```bash
python -m experiments.agents screen HARBOR_JOB_DIR/runs.jsonl \
  --minimum-success-rate 0.30 --maximum-success-rate 0.80 \
  --minimum-runs 3 --output HARBOR_JOB_DIR/admission.json
```

`eligible: false` is a hard stop for an efficacy sweep. Partial verifier checks
remain useful diagnostics, but do not override zero official task success.

## Required controls

Every compared condition uses identical task IDs and a fresh worktree. No PRA
preserves the agent's native compaction, cache, and tool policy. Native Memory
is `NOT_APPLICABLE` unless a real engine capability check succeeds. A selected
condition that saves tokens but solves fewer tasks is not better.
