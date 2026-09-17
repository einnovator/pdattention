# Task 5 recent-frontier DAG M=2 repeat 2

This is the second autonomous execution forked from the same frozen
four-episode persistent-FULL prefix as `autonomous_task5_m2_v1`.

- instance: `django__django-16145`
- official outcome: resolved
- calls: 12
- cumulative materialized input: 138,570 message-content tokens
- historical paired FULL input: 264,204 tokens in 12 calls
- paired saving: 47.55%
- own-trajectory full counterfactual: 267,078 tokens
- own-trajectory pruning: 48.12%

Together, the two M=2 executions resolve 2/2 with 12 calls each and paired
savings of 48.13% and 47.55%. They are repeated executions of one task
identity, not two independent accuracy observations.

`autonomous_metrics.json`, `official_result.json`, `run_manifest.json`, and
`trajectory.json` are copied verbatim from the immutable remote run directory.
