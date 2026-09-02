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
