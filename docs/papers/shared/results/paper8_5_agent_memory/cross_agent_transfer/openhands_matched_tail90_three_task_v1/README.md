# OpenHands three-task matched-tail cohort

This cohort is the first cross-agent autonomous qualification of the frozen
Paper 8.5 matched causal-tail policy. The three tasks were admitted only after
independent FULL resolution by the same OpenHands 1.49.2 agent and
`qwen3-coder:30b` model. Native condensation is disabled, temperature is zero,
request timeout is pinned to 1,200 seconds, retries are zero, and every action
has a delivered execution receipt.

Both FULL and selective arms resolve all three tasks. The candidate uses 79
actions versus 114 for FULL. Within the candidate trajectories, whole-record
selection saves 20.71% of logical history and ordinary-text materialization
saves 22.49%. Relative to the paired FULL workloads, it saves 45.64% logical
input and 46.86% materialized input. Provider accounting independently falls
45.08% for prompt plus completion tokens. These are pooled workload ratios;
the task-macro materialized paired saving is 43.70%.

The result meets the exploratory 30--50% paired-saving target with zero
observed resolution loss and no aggregate call increase. It is not yet a
production-default claim: the cohort has only three task identities, one
agent, one model family, and one run per selective cell. Task-level effects are
heterogeneous. In particular, Task 4 obtains most paired saving from a shorter
trajectory, while its whole-record own saving is only 1.94%.

`cohort.json` is the authoritative compact reduction and `cohort.csv` provides
the task-level table. Raw traces and official reports remain in the three
task-specific sibling directories.
