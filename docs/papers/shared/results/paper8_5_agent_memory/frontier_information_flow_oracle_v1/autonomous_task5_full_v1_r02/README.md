# Task 5 contemporaneous FULL repeat

This is the contemporaneous FULL control for the third clean M2+P1 execution
on `django__django-16145`.

- official outcome: not resolved
- calls: 10
- cumulative materialized input: 211,898 message-content tokens
- logical/materialized retention: 100%

The historical FULL execution resolved in 12 calls and sent 264,204 tokens.
The changed official outcome is direct evidence that temperature zero and a
fixed seed do not make this backend trajectory repeatable. Because this control
fails, its candidate pair cannot be counted as preservation of a FULL success.

The metrics, official result, manifest, request-selection ledger, and
trajectory are copied verbatim from the immutable remote run directory.
