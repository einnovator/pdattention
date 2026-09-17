# Task 4 contemporaneous FULL repeat

This is the matched contemporaneous FULL control for the second clean M2+P1
execution on `pytest-dev__pytest-7982`.

- official outcome: resolved
- calls: 7
- cumulative materialized input: 121,856 message-content tokens
- logical/materialized retention: 100%

This closely reproduces the historical FULL control (7 calls and 121,983
tokens). It is the denominator for the contemporaneous M2+P1 repeat.

The metrics, official result, manifest, request-selection ledger, and
trajectory are copied verbatim from the immutable remote run directory.
