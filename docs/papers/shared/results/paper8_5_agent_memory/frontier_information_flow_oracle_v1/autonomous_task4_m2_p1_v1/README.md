# Task 4 recent-frontier DAG M=2 plus P1

- instance: `pytest-dev__pytest-7982`
- official outcome: resolved
- calls: 6, versus 7 for historical paired FULL
- cumulative materialized input: 49,667 message-content tokens
- historical paired FULL input: 121,983 tokens
- paired saving: 59.28%
- own-trajectory full counterfactual: 104,705 tokens
- own-trajectory pruning: 52.56%

The most recent valid protocol completion is already inside the M=2 frontier,
so P1 is a logical no-op: every selected-message digest and metric matches the
M2-only Task 4 execution. This is the expected control for the targeted P1
repair.

The JSON metrics, official result, manifest, and trajectory are copied
verbatim from the immutable remote run directory.
