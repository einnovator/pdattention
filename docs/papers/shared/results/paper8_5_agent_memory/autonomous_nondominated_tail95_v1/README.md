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

## Same-host repeated qualification

The cross-host limitation above was resolved by adding contemporaneous FULL
controls on the same agent/Docker host as each treatment. The qualified cohort
contains three task identities and two runs per arm:

| Task | FULL primary | FULL calls | Tail-95 primary | Tail-95 calls | Tail gross saving |
|---|---:|---:|---:|---:|---:|
| `django__django-15277` | 1/2 | 40, 27 | 2/2 | 31, 15 | 6.05% |
| `pytest-dev__pytest-7982` | 2/2 | 21, 23 | 2/2 | 12, 17 | 10.17% |
| `scikit-learn__scikit-learn-13135` | 2/2 | 20, 23 | 0/2 | 19, 19 | 4.86% |

The pooled treatment saves 6.35% of its own cumulative materialized message
tokens and reduces mean calls from 25.67 to 18.83. Primary task-macro
resolution is 83.3% for FULL and 66.7% for tail-95. The treatment therefore
passes the 2% yield gate but fails the predeclared 80% accuracy gate and is
stopped. The large paired token reduction remains diagnostic because the FULL
repeats are not exact trajectories; failure-aware accounting is used so a
short failed run cannot appear efficient.

Both scikit-learn treatment repeats are exact reproductions. They make the
intended one-line source edit, but the completion command returns a source
excerpt rather than the required unified diff. SWE-bench consequently reports
`Patch Apply Failed`. This is a submission/progress-spine failure, not evidence
that the code edit was wrong. The workspace-capability outcome remains unknown
because the auxiliary checkpoint was unavailable; it does not replace the
primary 0/2 result.

`task05-full-a-contaminated` preserves an official FULL solve that is excluded
from reduction: pausing it behind another model stream left request indexes
1--11 and 13--22. The reducer correctly rejected the sparse trace. A clean
FULL-C was run instead. `qualification_decision.json` records the gate, while
`qualified_three_task_curves` contains the twelve-run reduction and plots.
