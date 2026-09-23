# OpenCode Task-1 watchdog-qualified FULL control

OpenCode 1.18.31 with the locked Qwen3-Coder-30B 64K alias again produced the
minimal 689-byte patch for `django__django-15277`; the independent SWE-bench
grader resolved the instance (1/1).

This fresh execution exercised the declared native-stop watchdog.  OpenCode
emitted `step_finish(reason=stop)` but did not exit during the 30-second grace
period, so the controller stopped only the post-completion process.  The
manifest therefore reports both `native_stop_observed=true` and
`forced_stop_after_native_stop=true`; exit 137 is lifecycle telemetry after
semantic completion, not a task failure.

The complete root trace has 16 model calls and 15 native tool calls (six Bash,
four read, two grep, two task-progress updates, and one edit), no nested agent,
and no invalid JSON.  It reports 154,751 cumulative logical input tokens,
143,170 cache-read tokens, 11,581 uncached input tokens, and 994 output tokens.
Absolute token totals are not pooled across agents because native prompts and
tool schemas differ.

This is a FULL capability/control row.  No PRA retention policy was applied.
