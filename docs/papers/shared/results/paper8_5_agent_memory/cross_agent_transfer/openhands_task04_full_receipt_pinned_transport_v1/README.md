# OpenHands Task 4 FULL control with pinned transport

This immutable artifact completes OpenHands' predeclared plain-three admission
cohort. It uses OpenHands SDK 1.49.2, `qwen3-coder:30b`, temperature zero,
disabled native condensation, FULL logical history, a 1,200-second per-request
timeout, and zero automatic retries on `django__django-15741`.

All 43 model requests succeeded. The agent emitted and delivered 43 execution
receipts, finished normally in 1,623.2 seconds, and submitted a 1,049-byte
patch. The trajectory materializes 227,056 cumulative message-content tokens;
provider accounting reports 812,507 prompt and 7,130 completion tokens. At the
last generation, 42 prior receipts are joined and 14 file observations have
exact resource versions. The 27 generic terminal actions remain unknown-effect
barriers.

The official SWE-bench 4.1.0 grader resolves the task: 2/2 FAIL_TO_PASS and
104/104 PASS_TO_PASS tests succeed. Together with Tasks 1 and 2, OpenHands now
has a 3/3 FULL admission cohort. All three identities may enter paired policy
evaluation. This artifact itself does not qualify a selective policy.

`qualification.json` is the compact evidence ledger. `openhands_events.jsonl`
contains native events and receipts; `proxy_trace.jsonl` records the complete
ordinary-text request audit and provider usage; the official reports are the
authoritative quality outcome.
