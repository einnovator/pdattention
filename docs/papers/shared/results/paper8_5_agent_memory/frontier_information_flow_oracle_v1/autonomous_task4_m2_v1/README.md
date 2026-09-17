# Task 4 recent-frontier DAG M=2

This autonomous execution is forked from the exact saved three-episode
persistent-FULL prefix. The logical policy is unchanged from the Task 5
pilots.

- instance: `pytest-dev__pytest-7982`
- official outcome: resolved
- calls: 6, versus 7 for historical paired FULL
- cumulative materialized input: 49,667 message-content tokens
- historical paired FULL input: 121,983 tokens
- paired saving: 59.28%
- own-trajectory full counterfactual: 104,705 tokens
- own-trajectory pruning: 52.56%

This is the second distinct paired-FULL-success task identity. Its saving is
above the predeclared 40--50% target band, so the unchanged third-identity run
is needed before aggregating or selecting a profile.

`autonomous_metrics.json`, `official_result.json`, `run_manifest.json`, and
`trajectory.json` are copied verbatim from the immutable remote run directory.
