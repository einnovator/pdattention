# Task 5 recent-frontier DAG M=2 plus P1 clean retry

- instance: `django__django-16145`
- official outcome: resolved
- calls: 12, equal to historical paired FULL
- cumulative materialized input: 138,570 message-content tokens
- historical paired FULL input: 264,204 tokens
- paired saving: 47.55%
- own-trajectory full counterfactual: 267,078 tokens
- own-trajectory pruning: 48.12%

The selected messages, model responses, commands, metrics, patch, and official
outcome exactly reproduce the earlier M2-only repeat. P1 is a logical no-op on
this task because a valid protocol completion is already inside the M=2
frontier.

The JSON metrics, official result, manifest, and trajectory are copied
verbatim from the immutable clean remote run directory.
