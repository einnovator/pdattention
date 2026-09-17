# Strict exact-prefix M2/P1 cohort

This ledger joins only treatments whose exact frozen-prefix FULL control ran
first and resolved officially. It does not include failed FULL qualification
attempts as treatment failures, nor does it reuse historical controls.

| Task | Episode | FULL solve/calls/tokens | M2/P1 solve/calls/materialized | Candidate own saving | Paired saving |
|---|---:|---:|---:|---:|---:|
| Task 3 | 3 | 1 / 7 / 103,713 | 1 / 7 / 92,434 | 11.52% | 10.88% |
| Task 4 | 4 | 1 / 14 / 260,662 | 1 / 6 / 49,667 | 52.56% | 80.95% |
| Task 5 | 5 | 1 / 9 / 191,697 | 1 / 9 / 97,780 | 49.64% | 48.99% |
| Aggregate | - | 3/3 / 30 / 556,072 | 3/3 / 22 / 239,881 | 40.53% | 56.86% |

The candidate-own denominator is the primary logical-selector quantity because
it compares selected and full-counterfactual history along the same treatment
trajectory. The paired denominator is the task-level workload quantity; it also
captures changes in calls-to-solution. Both matter, but they answer different
questions.

The observed preservation rate is 3/3, not a claim of greater-than-90%
population accuracy. The cohort is too small and the endpoint is not perfectly
repeatable. The result qualifies M2/P1 as the leading promotion candidate and
justifies additional task identities; it does not yet justify a production
default or Easy-14 accuracy claim.

See `evidence.json` for machine-readable totals and links to each task bundle.
