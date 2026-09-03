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
