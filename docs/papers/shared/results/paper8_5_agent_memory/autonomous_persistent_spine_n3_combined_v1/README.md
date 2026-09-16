# Combined persistent N=3 reliability evidence

This bundle reduces four immutable campaigns over the same ordered
Django--Pytest--Scikit-learn sequence family. Repeated executions are clustered
by ordered sequence family; they are not treated as independent task samples.

| Policy | Executions | Official resolutions | Mean failure-aware saving | Mean call delta | Gate |
|---|---:|---:|---:|---:|---|
| persistent FULL | 6 | 18/18 | 0.00% | 0.0 | control |
| R4/M2/V2 | 2 | 4/6 | 0.00% | -6.0 | reject: lost two successes |
| R6/M2/V2 | 2 | 5/6 | -1.52% | +1.5 | reject: one lost success and mean call increase |
| R8/M2/V2 | 1 | 3/3 | 13.09% | +6.0 | reject: below saving target and call increase |
| R4/M2/V2/P1 | 2 | 6/6 | 41.11% | -7.5 | repeated discovery gate pass |

The R6 executions are deliberately not collapsed into a favorable aggregate:
one resolves 3/3 but consumes 3.03% more tokens than paired FULL and adds ten
calls; the replication saves 37.56% raw tokens and seven calls but resolves
only 2/3, so its predeclared failure-aware saving is zero. No tested policy
before P1 meets the 30--50% saving, zero-loss, no-call-increase gate.

R4/M2/V2/P1 adds one clean completed action--observation protocol exemplar per
prior issue. Its two independent executions both resolve 3/3, save 36.41% and
45.82% against their contemporaneous FULL controls, and reduce calls by six
and nine. The clustered descriptive mean is 41.11% saving and 7.5 fewer calls.
This is a repeated discovery result on one ordered sequence family, not yet a
held-out-sequence or population-level confirmation.

Files:

- `reliability_frontier_runs.jsonl`: R4, first R6, R8, and their controls.
- `r6_replication_frontier_runs.jsonl`: independent R6 replication and control.
- `protocol_v1_frontier_runs.jsonl`: first P1 execution and control.
- `protocol_replication_frontier_runs.jsonl`: independent P1 replication and control.
- `persistent_frontier_reduction.json`: strict paired reduction.
- `curves/curve_summary.json`: ordered-sequence-family-clustered descriptive summary.
- `curves/`: regenerated quality, call, and divergence plots.
