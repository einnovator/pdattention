# Longer-horizon FULL control: Task 1

This is the first clean execution from the predeclared Easy-50 horizon cohort.
It runs `sympy__sympy-17655` with mini-swe-agent 2.4.6, the frozen
`qwen3-coder:30b` revision, deterministic generation, FULL history, and an
80-action ceiling.

The agent submitted after 60 model calls, ten calls beyond the original
50-call limit. It did **not** resolve the task. The final submission was prose
rather than a source patch, so the official grader classified it as
`patch_apply_failed`. The run consumed 946,069 cumulative logical input tokens
and 15,946 reported completion tokens over 943.3 seconds. An intermediate
16-line edit was later reverted; because no source-only patch survived at
submission, this identity does not satisfy the predeclared forward-progress
gate for a 120-call continuation.

This result is capability evidence only. It does not change the Easy-14 policy
denominator and is not a memory-policy failure because no selection was
applied.

Files:

- `run_manifest.json`: frozen environment and invocation declaration.
- `autonomous_metrics.json`: call and token accounting.
- `official_result.json`: primary official outcome.
- `preds.json`: submitted model patch payload.
- `exit_statuses.yaml`: agent termination state.
- `persistent_episode_export.json`: normalized 60-action history.
- `request_selection.jsonl`: per-request FULL-selection ledger.
- `grader.log`: official grading trace.

