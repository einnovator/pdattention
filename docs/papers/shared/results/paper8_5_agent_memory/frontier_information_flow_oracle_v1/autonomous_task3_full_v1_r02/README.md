# Task 3 contemporaneous FULL repeat

This is the matched contemporaneous FULL control for the second clean M2+P1
execution on `scikit-learn__scikit-learn-12585`.

- official outcome: resolved
- calls: 13
- cumulative materialized input: 199,232 message-content tokens
- logical/materialized retention: 100%

The historical FULL execution resolved in six calls and sent 88,510 tokens.
This repeat therefore demonstrates large endpoint trajectory variance even
when no history is removed. It is the valid denominator for the contemporaneous
M2+P1 repeat, which also resolves in 13 calls and sends 179,380 tokens (9.96%
paired saving).

The metrics, official result, manifest, request-selection ledger, and
trajectory are copied verbatim from the immutable remote run directory.
