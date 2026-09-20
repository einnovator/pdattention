# Kilo Task-4 controlled FULL admission

Kilo 7.7.5 ran locked Easy-14 Task 4, `django__django-15741`, against the
same frozen `qwen3-coder:30b` revision and generation ceiling used by the
other transfer agents. The logical proxy passed all 13 requests through at
FULL retention. External telemetry, default plugins, project configuration,
and the persistent session database were disabled; this hermetic retry did
not reproduce the external TLS stall seen in the quarantined diagnostic.

The model correctly localized the lazy `format_type` failure, but its final
response reached the 1,024-token completion ceiling while describing the fix
and before issuing a write tool. Kilo treated that provider stop as terminal.
The workspace patch is empty, so the official SWE-bench report records an
unresolved empty-patch submission without executing tests.

The run made 13 model calls and 12 native tool calls and consumed 237,135
cumulative logical input tokens plus 1,836 output tokens. This is a genuine
FULL capability failure, not a memory-policy failure. Kilo therefore completes
the plain-three admission at 2/3; only Tasks 1 and 2 may enter Kilo policy
comparisons.

`aggregate_report.json` is the official aggregate report. `proxy_trace.jsonl`
preserves the exact FULL request audit, while `conversation_events.jsonl` and
`event_summary.json` preserve the native Kilo trajectory.
