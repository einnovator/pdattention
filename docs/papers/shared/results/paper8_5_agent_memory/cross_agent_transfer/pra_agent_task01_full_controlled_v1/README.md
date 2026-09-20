# PRA Agent Task-1 FULL admission

This directory contains the corrected same-model FULL admission for PRA Agent
on locked Easy-14 Task 1, `django__django-15277`, using `qwen3-coder:30b`.

The official SWE-bench instance report resolves the issue: 2/2
FAIL_TO_PASS and 158/158 PASS_TO_PASS tests pass. The agent makes 14 model
requests and 13 typed tool calls and leaves only the minimal 689-byte source
patch. Provider usage is 125,952 cumulative prompt tokens, including 27,362
cached prompt tokens, and 1,725 completion tokens. These are agent-specific
FULL-control measurements; they are not comparable to another agent's raw
token total.

The first frozen attempt is retained under
`diagnostic_pre_fix_text_action/`. It stopped after 10 requests and nine tool
calls with an empty patch because the model serialized an otherwise valid
`search_text` action as the PRA-owned durable text projection rather than as a
native `tool_calls` object. Commit `c61fe804` accepts that representation only
when the exact action sentence terminates the response. The corrected retry
continued from the same point, produced the minimal fix, verified it, and
resolved officially.

The official harness completed the instance and wrote
`official_instance_report.json`. Its aggregate-report writer subsequently hit
a Docker container-list race, so `official_report.json` is an explicitly
derived one-instance index, not a replacement grade. The per-instance report
and test output are authoritative.

This admission supplies compatibility evidence only. It does not qualify a
retention policy and does not make a native-K/V claim.
