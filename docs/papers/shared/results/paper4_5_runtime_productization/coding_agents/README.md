# Coding-agent campaign artifacts

These artifacts establish the reproducible campaign surface described in
`experiments/agents/`. The fixture runs validate fresh-workspace isolation,
normalized JSONL, metric aggregation, and paired analysis. They are not model,
agent, Terminal-Bench, or SWE-bench quality evidence.

`local_agent_audit.json` records a non-invasive Windows-host executable audit.
An absent executable means only that the agent was not installed on that host.
`terminal_bench_smoke_plan.json` and `swebench_lite_smoke_plan.json` delegate to
the official harnesses and include every frozen task filter explicitly.

No performance claim should be made until the corresponding official harness
produces normalized per-task records on a qualified agent/engine combination.

## Official harness qualification

Harbor 0.22.0 executed the `terminal-bench/filter-js-from-html` oracle on the
NVIDIA/WSL host against Terminal-Bench 2.1. The official verifier awarded
`1.0` (one trial, zero exceptions). The preserved
`terminal_bench_oracle_smoke/` directory contains the resolved job and trial
configuration, locks, oracle transcript, verifier output, and result records.
This validates the downloaded task and official grading path; it is an oracle
qualification, not model-agent evidence.

## First Terminal-Bench model qualification

OpenCode 1.18.26 with Ollama/Qwen3-14B then ran the five frozen smoke tasks
through the PRA Gateway in No-PRA mode. Harbor completed all five trials with
zero harness exceptions, but the official task-success result was `0/5`.
Verifier-level tests were `7/12`: `filter-js-from-html` 1/2,
`configure-git-webserver` 0/1, `fix-git` 1/2, `polyglot-c-py` 0/1, and
`query-optimize` 5/6. The agent reported 100,883 cumulative input tokens.

This is valid negative model-agent evidence, not a PRA comparison. The general
Qwen3-14B pairing is below the Stage-B promotion floor, so profile sweeps would
be uninformative. A code-specialized model must first establish nonzero task
success. The raw Harbor jobs, trajectories, verifier records, normalized
JSONL, and combined summary are preserved beside this README.

The canonical three-condition record is
`qwen3_14b_canonical_evidence.json`. It retains the measured No-PRA values and
marks PRA - No Adaptor and PRA - Adaptor Bundle as `BLOCKED`, not as numeric
zeroes. `qwen3_coder_30b_canonical_evidence.json` applies the same contract to
the one-task coder gate. Both records include the executable admission result
and computed-delta slots for downstream CLI, documentation, and Control Plane
renderers.

A follow-up OpenCode gate used the newly downloaded
Ollama/Qwen2.5-Coder-7B on `query-optimize` and `fix-git`. Both official tasks
failed (`2/6` and `0/2` verifier tests); the model emitted intended tool calls
as plain text, so OpenCode made no tool invocations. This pairing is also not
promoted. The subsequent controlled gate changed the agent loop while retaining
the tool-capable Qwen3-14B model.

A stronger code-model gate then ran OpenCode with Ollama/Qwen3-Coder-30B on
`query-optimize`. It also received official reward zero, while passing `4/6`
verifier tests. Unlike the 7B pairing, it exercised the tool path: six model
calls produced five tool calls, one read, and four writes. However, the final
SQL retained a correlated subquery. The official performance verifier therefore
ran to its 30-minute boundary; Harbor completed without a harness exception at
1,791.2 seconds total wall time. The run used 50,876 cumulative input tokens.
This is a model/task failure rather than gateway or tool-transport failure, and
the 30B pairing is not promoted to a PRA profile sweep.

That same-model gate is now complete for `query-optimize`. OpenCode passed
`5/6` verifier tests, Pi 0.73.1 passed `4/6`, and OpenHands 0.57.0 passed
`2/6`; all three received official task reward zero. Pi made three model calls
and executed one file read plus one file write, using 5,219 cumulative input
tokens. OpenHands entered its continuation loop without producing `sol.sql`;
Harbor's OpenHands adapter had to be pinned to 0.57.0 because version 1.11.0 no
longer exposes the module path expected by Harbor 0.22.0. These one-task rows
are compatibility diagnostics, not agent rankings.

The gateway also now pins the configured backend model when clients send a
provider-qualified model name such as `openai/qwen3:14b`. A live request still
succeeded after the temporary provider-qualified Ollama alias was removed,
which confirms that model-name translation occurs at the gateway boundary.
The compact cross-agent comparison is in
`terminal_bench_qwen14b_cross_agent_gate_summary.json`.
The normalized 30B row and all official artifacts are under
`terminal_bench_opencode_qwen3coder30b_gate/`.

## Larger stronger-model cohort

The preregistered `qwen3_coder_30b_multi_harness_v3` campaign expands the
stronger-model qualification to ten frozen Terminal-Bench 2.1 tasks and three
agent harnesses. Ollama/Qwen3-Coder-30B Q4_K_M ran on an Apple M4 Pro with
48 GiB of unified memory; Harbor 0.22.0 and the official task verifiers ran on
a separate 16 GiB Apple host. Every cell was No-PRA.

All 30 official trials completed. Mini-SWE-agent 2.4.6 and Qwen Code 0.23.0
each solved `fix-git`; Aider 0.86.2 solved none. The resulting 2/30 success
rate is 6.7% (Wilson 95% interval 1.8--21.3%), and only one of ten unique tasks
was solved by any harness. The configured 10% promotion floor therefore keeps
all PRA treatments blocked. This is stronger evidence than the earlier
one-task diagnostic, but it is still an admission result rather than a PRA
comparison or an agent ranking.

Mini-SWE-agent reported 2,562,781 input and 48,888 output tokens over ten
trials; Qwen Code reported 4,154,315 input and 38,821 output tokens. Together,
the 20 token-reporting trials accumulated 6,717,096 input tokens and 87,709
output tokens. Aider's Harbor integration did not expose token or turn
telemetry, so those fields are recorded as not reported rather than interpreted
as zero consumption. Aggregate task wall time was 4.02 hours across the 30
trials. Six agent timeouts and one verifier timeout are retained as model/harness
outcomes because usable trajectories preceded them.

The directory contains the normalized `runs.jsonl`, durable matrix state,
summary, and report. `raw_harbor_artifacts.tar.gz` preserves the complete
campaign tree, including task receipts, trajectories, logs, and verifier
outputs. The archive is used because the original Harbor nesting exceeds the
legacy Windows path limit when checked out file by file.

## Stage A gateway qualification

On 2026-09-02, Codex CLI 0.147.0, OpenCode 1.18.26, and Pi 0.73.1 each
completed the two exact-output fixture tasks twice through the PRA Gateway and
Ollama/Qwen3-14B on the M5 host. All 12 runs succeeded. Codex used the Responses
protocol; OpenCode and Pi used Chat Completions. The runs validate streaming,
tool calls, isolated workspaces, and normalized artifacts only.

| Agent | Runs | Success | Input tokens | Mean task time |
| --- | ---: | ---: | ---: | ---: |
| Codex CLI | 4 | 4/4 | 97,958 | 47.3 s |
| OpenCode | 4 | 4/4 | 53,170 | 107.7 s |
| Pi | 4 | 4/4 | 14,482 | 44.0 s |

The token totals are agent-reported cumulative inputs and are not directly
comparable as model-efficiency measurements: each harness contributes a
different system prompt and tool schema. The fixture is too small for latency
or quality claims. It did reveal and motivate fixes for decode-limit
forwarding, streamed tool calls and usage, `tool_calls` finish reasons,
BOM-marked Windows files, provider persistence, and large transcript storage.

## Easy-50 gateway invalidation

The admitted Qwen3-Coder-30B No-PRA Easy-50 baseline resolved `14/50`. Its
first nominal G00 result was `0/50`, but all 50 trajectories ended on their
first API call with `InternalServerError` and no submission. This is an invalid
gateway transport run, not a model or PRA result. The retained raw normalized
rows and explicit machine-readable invalidation live under
`swebench_verified_easy50/gateway_passthrough_invalid_transport/`.

The repair preserves ordinary OpenAI request fields and upstream response
metadata, accepts backend origins and `/v1` API roots, pins the downstream
model, returns structured upstream failures, and probes a real generation
before agent launch. A three-task direct-solved parity cohort must pass before
the corrected Easy-50 condition is admitted.
