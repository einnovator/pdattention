# Task 5 recent-frontier DAG M=2 plus P1 third execution

This is the third clean autonomous execution of the frozen M2+P1 policy on
`django__django-16145`.

- official outcome: resolved
- calls: 12
- cumulative materialized input: 138,570 message-content tokens
- contemporaneous FULL outcome: not resolved
- contemporaneous FULL calls/input: 10 / 211,898 tokens
- descriptive paired input reduction: 34.61%
- own-trajectory full counterfactual: 267,078 tokens
- own-trajectory pruning: 48.12%

The candidate exactly reproduces the preceding successful M2+P1 execution's
12 calls and 138,570 tokens and again resolves officially. The contemporaneous
FULL control fails, so this is not a paired-success preservation point and does
not enter the conditional saving gate. It is unconditional evidence that
retiring disconnected history can remove harmful cross-task interference, a
hypothesis that requires randomized repeated pairs.

The metrics, official result, manifest, request-selection ledger, and
trajectory are copied verbatim from the immutable remote run directory.
