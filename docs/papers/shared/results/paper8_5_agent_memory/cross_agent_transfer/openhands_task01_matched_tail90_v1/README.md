# OpenHands Task 1 matched causal-tail qualification

This immutable artifact pairs the qualified OpenHands FULL control with a
prospective `matched_token_tail` run on `django__django-15277`. Both executions
use OpenHands SDK 1.49.2, `qwen3-coder:30b`, temperature zero, disabled native
condensation, the same task-derived workspace, and the ordinary OpenAI-tools
protocol with complete live execution-receipt delivery.

The selective run is officially resolved: 2/2 FAIL_TO_PASS and 158/158
PASS_TO_PASS tests pass. It finishes in 24 actions rather than the FULL
control's 33. Across its own trajectory it exposes 147,757 full-history tokens
to selection and materializes 84,919, a 42.53% own saving. Relative to the
FULL trajectory's 188,945 cumulative message-content tokens, it materializes
55.06% fewer tokens. The paired figure includes the beneficial nine-action
trajectory contraction and must therefore remain separate from own saving.

The nominal 90% setting is a strict maximum, not a retention target. Because
the selector admits complete causal turns and pins the system prompt, task
statement, and current turn, indivisible older turns leave unused budget. The
observed request-level retention ranges from 42.61% to 100%, and the last
request retains 55.25%. This is why the paper reports realized retention and
budget underfill rather than labeling the outcome as "90% retained."

`proxy_trace.jsonl` is the per-request selection and usage ledger.
`run_manifest.json`, `event_summary.json`, and `openhands_events.jsonl` preserve
the native run. `official_report.json` and `official_instance_report.json`
contain independent SWE-bench grading evidence. `qualification.json` records
the compact paired result.
