# Codex native-Ollama Task-1 FULL admission

Codex CLI 0.153.0, using its native Ollama OSS provider and the locked
Qwen3-Coder-30B 64K serving alias, produced the established minimal 689-byte
patch for `django__django-15277`.  The independent SWE-bench grader resolved
the instance (1/1).

The run completed normally in 399.6 seconds.  Its native JSON event stream has
80 events: 24 agent messages, 52 command-execution start/completion events,
one provider error event, and a terminal `turn.completed`.  Final native usage
reports 397,455 cumulative input tokens, of which 380,392 were cache reads,
and 4,045 output tokens.

This is a capability and transport-admission result, not a frozen-sampling or
PRA-selection result.  Codex's native Ollama mode does not expose enforceable
temperature, top-p, or seed controls, and this FULL execution did not apply a
retention policy.  The remote-host support is a transparent loopback TCP
bridge: Codex still owns the Ollama protocol and agent behavior.

Evidence:

- `run_manifest.json`: immutable treatment and observed model identity;
- `event_summary.json`: normalized native-event and usage counts;
- `codex_events.jsonl`: complete native event stream;
- `model.patch`: submitted source patch;
- `official_grade.json`: independent SWE-bench resolution report.
