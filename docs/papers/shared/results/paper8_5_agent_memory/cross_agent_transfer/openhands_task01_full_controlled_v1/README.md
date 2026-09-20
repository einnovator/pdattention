# OpenHands Task-1 controlled FULL admission

OpenHands SDK 1.49.2 ran locked Easy-14 Task 1,
`django__django-15277`, inside the official task-derived container with its
native terminal, file-editor, task-tracker, think, and finish actions. The SDK
condenser was explicitly disabled. External telemetry was disabled and an SDK
telemetry defect was guarded by treating an absent optional
`cache_creation_tokens` usage counter as zero; that repair changes accounting
only, not prompts, responses, tools, or selection.

The clean immutable control contains 33 contiguous model requests. Every
request is exact FULL pass-through at 100% retention. OpenHands produces the
minimal 689-byte guarded `MaxLengthValidator` patch, and the official
SWE-bench 4.1.0 grader resolves the task: 2/2 FAIL_TO_PASS and 158/158
PASS_TO_PASS tests pass.

The run consumes 641,509 cumulative provider prompt tokens and 5,747
completion tokens. Its native trajectory contains 33 actions: 18 terminal,
13 file-editor, one think, and one finish action. Four observations carry
semantic failure evidence. These totals are valid only for within-OpenHands
policy pairing because its system prompt and tool schemas differ from the
other agents.

An earlier post-fix diagnostic also produced the identical source patch but
used 35 actions. The first 21 assistant contents agree before divergence,
despite temperature zero and identical request parameters. That diagnostic is
not pooled with the clean control; it demonstrates why exact trajectory
repeatability is auxiliary rather than the cross-agent quality endpoint.

`official_report.json` and `official_instance_report.json` contain official
grading evidence. `proxy_trace.jsonl` is the request/usage/retention audit,
while `openhands_events.jsonl` and `event_summary.json` preserve native events.
`portable_execution_receipts_v1.json` is the agent-edge translation into the
generic PRA outcome/resource contract. It pairs all 33 actions with results,
separates four semantic failures from successful transport, and marks one
clipped observation incomplete. Only two non-workspace control operations have
complete effect traces; 31 terminal/editor operations correctly remain
unknown-effect barriers because this historical run did not capture filesystem
versions at execution time. This is a contract smoke, not a retention-policy
result.
