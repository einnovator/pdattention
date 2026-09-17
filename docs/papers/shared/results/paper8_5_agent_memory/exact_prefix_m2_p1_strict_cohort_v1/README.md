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

## Frozen Paper 4.5 handoff

`export_paper4_5_frozen_plan.py` reconstructs every canonical pre-selection
request from the immutable prefix and trajectory, verifies its request and
selected-message hashes, and translates the already selected records into the
common Paper 4.5 resource fixture.  It does not rerun or approximate M2/P1.
The current-episode mandatory floor is exported explicitly so a downstream
engine harness cannot pull the preceding issue's final turn back into a new
issue.

| Task | Requests | Frozen fixture SHA-256 | Frozen request replay SHA-256 |
|---|---:|---|---|
| 3 | 7 | `04c9fbad18b9a98648da86edecf8a88fe40dc915a1e58e75154ab472dcd2f1e5` | `ff3f0983ec2d4a8b086ab5631eb8fc331fdef8f03073bd0b5f70887bac3e2ff2` |
| 4 | 6 | `b1f0b05f3266d2d3cf2dd18c3d308da90898aedd9d4b7909b3a6f4e80d8bf777` | `12f42421cdbb925e82335753fe89f98fba453bdc646e1ed5909b7a199bd06be9` |
| 5 | 9 | `e292c5e255a17d4b77f354f09a73cedbae2e9172d965872811f537d0cf9dfc82` | `86de3ea93f477940f8f041bf6bf378cd7a80ce93e69932b68f8a957957a2941d` |

Each `paper4_5_task*_request_replay.jsonl` file contains the exact full logical
messages, frozen generation settings, and recorded assistant response for the
same decisions.  The adjacent manifests bind both files to the prefix,
trajectory, selection trace, segmentation rule, source policy, model, and
request count.  Logical success and token saving remain Paper 8.5 claims.
Paper 4.5 must replay these fixtures unchanged and report resident-K/V
realization, re-encoding, copy, temporary bytes, lifecycle behavior, and task
behavior separately.
