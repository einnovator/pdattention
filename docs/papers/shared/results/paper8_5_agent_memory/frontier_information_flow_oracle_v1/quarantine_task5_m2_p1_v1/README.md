# Quarantined Task 5 M=2 plus P1 execution

This execution is retained for instability analysis but excluded from official
accuracy aggregation. Generation completed in 16 calls and produced a valid
git-diff envelope, but the patch accidentally moved/removes runserver control
flow. The official Django grader remained inside `admin_scripts.tests` for
more than 12 minutes versus about 77 seconds for paired FULL. The campaign
timeout rule stopped the grader and started a clean immutable retry.

- cumulative materialized input: 189,982 tokens
- historical paired FULL input: 264,204 tokens
- descriptive paired saving before grading: 28.09%
- call delta: +4
- official outcome: unavailable; never count this row as resolved or failed

The first four requests and commands match the successful M2 execution
exactly, and request 5 has an identical selected prompt and command but
different model reasoning text. Later trajectory divergence is therefore a
temperature-zero backend-repeatability observation, not a P1 selection
change. The incomplete run directory itself is not reused by the retry.
