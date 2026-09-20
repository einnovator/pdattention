# Pi Task-2 controlled FULL admission

This directory records the first exact-parameter cross-agent control for
Paper 8.5. Pi 0.75.3 used its native typed tools against
`qwen3-coder:30b` with temperature 0, top-p 1, seed 0, and a 1,024-token
completion ceiling. The compatibility proxy retained every request message
and preserved native `tool_calls` and `tool_call_id` fields.

The agent produced the minimal one-line Django fix. The official SWE-bench
4.1.0 per-instance report resolves the task: 1/1 FAIL_TO_PASS and 29/29
PASS_TO_PASS tests passed. The harness subsequently encountered a known
Docker stale-container race while constructing its aggregate report; this
occurred after the authoritative per-instance result was written.

The run made 22 model calls and 21 tool calls. Six tool results were errors,
including edit-schema mistakes and failed environment probes. This makes
error/recovery groups a required atomic unit for cross-agent policies. Pi's
reported 419,139 cumulative logical input tokens include its system prompt
and tool schemas, so they must not be pooled with mini-swe-agent's
message-content-only counts. Policy savings will be paired within Pi.

`qualification.json` is the compact decision record. `proxy_trace.jsonl`
audits FULL identity and fixed generation parameters. The 4.5 MB raw Pi
stream is kept outside Git at
`D:/git/rd/p85-pi-transfer-raw/pi_task02_full_controlled_v1_events.jsonl`;
its digest is recorded in `event_summary.json`.
