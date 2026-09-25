# Pi persistent-session N=3 evidence

This directory contains the first prospective Pi 0.75.3 persistent-session
comparison for the frozen boundary-free Recent-Frontier M2/P1 policy.

## Locked cohort

1. `django__django-15277`
2. `django__django-15368`
3. `scikit-learn__scikit-learn-13135`

Both arms use Qwen3-Coder-30B at temperature zero and share the exact completed
Task-1 FULL prefix. The candidate retains the last two user-instruction epochs,
one protocol exemplar, two protected head turns, and four protected tail turns.
Heuristic retirement is explicit and auditable when exact execution-receipt
sidecars are absent.

## Reduced result

| Outcome | FULL | M2/P1 |
|---|---:|---:|
| Official resolutions | 2/3 | 3/3 |
| Calls | 115 | 107 |
| Whole-history/counterfactual tokens | 2,847,310 | 2,554,440 |
| Materialized tokens | 2,847,310 | 2,042,277 |

- Candidate-own logical saving: **20.05%**.
- Paired saving against the different FULL trajectory: **28.27%**.
- Preservation conditional on FULL success: **2/2**, with zero lost successes.
- Task 3: **52.34%** candidate-own saving and 27 rather than 35 calls.

Task 3 is a selective-recovery diagnostic because FULL failed with an empty
patch after a Pi structured-edit recovery loop. It is not counted as a
paired-preservation win. This is one task-clustered repeat and does not establish
population accuracy or a production default.

`pair_evidence.json` is the authoritative compact ledger. The two campaign
directories contain request-level traces; the official-grade directories
contain SWE-bench reports and logs. The reducer records hashes for every source
artifact used in the comparison.
