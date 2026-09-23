# OpenCode Task-1 native-stop auxiliary admission

OpenCode 1.18.31 with the frozen Qwen3-Coder-30B 64K alias produced the
correct two-line Task-1 patch.  The independent SWE-bench grader resolved it.
The root typed-event trace is complete: 24 assistant calls, 23 native tool
calls, no nested-agent calls, and no invalid JSON lines.

The run is auxiliary because the CLI emitted its native terminal
`step_finish(reason=stop)` event but its primary process remained alive until
the experiment controller stopped the container.  Exit code 137 therefore
describes process lifecycle after semantic completion, not task failure.  The
runner now has a declared native-stop watchdog and records native stop,
graceful exit, and forced post-terminal stop separately.  A fresh run is still
required before the row is called clean process-lifecycle qualification.

