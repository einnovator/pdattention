# Repeated autonomous matched-recency pilot

This directory records two FULL and two matched-token-tail@95 executions of
`django__django-15277` with mini-swe-agent 2.4.6 and the pinned
`qwen3-coder:30b` Ollama artifact. Temperature is zero, top-p is one, and seed
is zero. The primary outcome is the official SWE-bench task result.

The observed arm-level point is encouraging but not a valid paired causal
comparison:

- FULL resolves 1/2 runs at 29 calls per run.
- Matched recency resolves 2/2 runs at 31 and 15 calls.
- Matched recency saves 6.05% of cumulative message-content tokens when the
  two runs are pooled by token count.
- The policy first changes a materialized request at call 3.

The agent hosts differ. The first `find | grep | head` command returns the same
five paths in a different order on `.6` and `.8`. Consequently the model's
second command differs before the policy has changed any request. The results
must therefore remain arm-level observations; they do not show that selection
caused the higher resolution rate or lower mean call count. Same-host controls
are the next qualification requirement.

The tail runs also exposed two harness defects, both fixed after preserving the
raw evidence:

1. A small mandatory current turn could exceed the nominal 95% ceiling and was
   rejected instead of reported as mandatory overflow (`ab560838`).
2. SWE-bench could persist a resolving per-instance report and then fail during
   aggregate Docker cleanup (`21924b03`). The raw runner error is retained as
   `official_result_runner_error.json`; the canonical result is reconstructed
   only from `official_per_instance_report.json`.

`comparison.csv` is the row-level reduction and `comparison.json` records the
pre-policy divergence audit. Each run directory contains the manifest,
trajectory, per-request selection trace, instrumentation receipts, model patch,
and official evidence available on the producing host.
