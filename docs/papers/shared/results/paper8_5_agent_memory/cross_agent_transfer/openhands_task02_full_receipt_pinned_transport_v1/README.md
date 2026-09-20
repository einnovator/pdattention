# OpenHands Task 2 FULL control with pinned transport

This immutable artifact is the second successful OpenHands admission control.
It uses OpenHands SDK 1.49.2, `qwen3-coder:30b`, temperature zero, disabled
native condensation, and FULL logical history on `django__django-15368`.
Request behavior is explicit: 1,200 seconds per request and zero automatic
retries.

All 38 model requests succeeded. The agent emitted and delivered one receipt
for each of 38 action--observation pairs, finished normally in 425.7 seconds,
and submitted a 671-byte patch. The trace contains 153,174 cumulative
message-content tokens, 548,704 provider prompt tokens including the OpenHands
prompt and tool schemas, and 5,986 completion tokens. At the last generation,
37 prior receipts are joined and 14 file observations have exact resource
versions. The 21 generic terminal actions remain unknown-effect barriers.

The official SWE-bench 4.1.0 grader resolves the task: 1/1 FAIL_TO_PASS and
29/29 PASS_TO_PASS tests succeed. Task 2 therefore enters OpenHands policy
pairing together with Task 1. This artifact qualifies a FULL control and live
receipt transport; it does not qualify a selective policy.

An earlier diagnostic attempt is excluded. OpenHands' implicit request timeout
started overlapping retries during one long model call. That attempt was
stopped, preserved under `/tmp/p85-openhands-receipt-live/task2_run3`, and is
not counted. Commit `1df67b5b` pins the timeout and disables hidden retries.

`qualification.json` is the compact evidence ledger. `openhands_events.jsonl`
contains native events and execution receipts; `proxy_trace.jsonl` records the
ordinary-text request audit and provider usage; the two official report files
are the authoritative quality outcome.
