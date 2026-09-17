# Task 3 recent-frontier DAG M=2 plus P1 repeat

This is the second clean autonomous execution of the frozen M2+P1 policy on
`scikit-learn__scikit-learn-12585`.

- official outcome: resolved
- calls: 13, equal to the contemporaneous FULL repeat
- cumulative materialized input: 179,380 message-content tokens
- contemporaneous FULL input: 199,232 tokens
- contemporaneous paired saving: 9.96%
- own-trajectory full counterfactual: 201,740 tokens
- own-trajectory pruning: 11.08%

P1's correctness repair repeats: the run submits a valid patch and resolves
officially. Its efficiency does not repeat. The first two selected request
inputs are byte-identical to the first repaired run. The first assistant
content and command also match; generation diverges on request 2 despite the
identical selected input and frozen temperature/top-p/seed fields. The new
trajectory performs extra source reads and submission construction before
resolving.

The contemporaneous FULL repeat also resolves in 13 calls and sends 199,232
tokens. The policy therefore does not add calls in the matched repeat and saves
9.96%, but it fails the 30--50% paired-saving gate. The historical six-call
FULL comparison would falsely attribute backend trajectory variance to the
policy.

The metrics, official result, manifest, request-selection ledger, and
trajectory are copied verbatim from the immutable remote run directory.
