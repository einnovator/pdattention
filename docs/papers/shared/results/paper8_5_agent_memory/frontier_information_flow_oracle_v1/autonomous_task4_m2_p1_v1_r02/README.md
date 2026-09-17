# Task 4 recent-frontier DAG M=2 plus P1 repeat

This is the second clean autonomous execution of the frozen M2+P1 policy on
`pytest-dev__pytest-7982`.

- official outcome: resolved
- calls: 6, versus 7 for contemporaneous FULL
- cumulative materialized input: 49,667 message-content tokens
- contemporaneous FULL input: 121,856 tokens
- contemporaneous paired saving: 59.24%
- own-trajectory full counterfactual: 104,705 tokens
- own-trajectory pruning: 52.56%

The candidate exactly reproduces the first M2+P1 run's six calls and 49,667
tokens. P1 remains a selection no-op on this history because the required valid
completion is already inside the M=2 frontier. The contemporaneous FULL repeat
also closely reproduces its historical control, so this is a repeat-qualified
high-saving task identity.

The metrics, official result, manifest, request-selection ledger, and
trajectory are copied verbatim from the immutable remote run directory.
