# Qwen2.5-Coder-14B progress-spine Task 1 diagnostic

This bundle records the first autonomous use of the corrected
`head_tail_progress_spine` policy.  The earlier run carrying a progress-spine
label actually used `head_tail_recency` and is quarantined from this claim.

All executions use mini-swe-agent 2.4.6, locked Easy-14 Task 1
`django__django-15277`, Qwen2.5-Coder-14B-Instruct-4bit revision
`29efdbab55a161237ab1e432a3abaf6c7ae2b477`, its exact tokenizer,
temperature zero, top-p one, seed zero, a 1,024-token completion ceiling, the
backtick scaffold, the strict unified-diff submission guard, and official
SWE-bench grading.

| Execution | Repo revision | Official solve | Calls | Full tokens | Materialized tokens | Own saving |
|---|---|---:|---:|---:|---:|---:|
| earlier FULL success | `d9ff6354` | 1 | 14 | 39,043 | 39,043 | 0.00% |
| corrected progress-spine@90 | `ea2c2264` | 1 | 14 | 39,043 | 38,424 | 1.59% |
| fresh FULL stability repeat | `ea2c2264` | 0 | 7 | 17,410 | 17,410 | 0.00% |

The corrected treatment reproduces all 14 assistant actions and the submitted
patch from the earlier successful FULL control exactly.  Its only selection
changes occur at requests 12--14, whose per-request retentions are 95.36%,
92.85%, and 93.53%.  Consequently, this is a positive implementation and
interaction-parity result but a low-yield savings point on a short task.

The same-revision FULL repeat is a required instability control, not an
alternative favorable denominator.  Its first three responses equal the
treatment exactly.  At request 4 the serialized request digest and 2,470-token
FULL input remain identical, no selection has occurred, but the model response
differs.  The FULL repeat then submits an incorrect patch after seven calls.
Thus the fork precedes selection by eight requests and demonstrates that this
MLX temperature-zero endpoint is not exactly repeatable.  The successful
treatment cannot be promoted as a paired policy-quality estimate from this
single identity.

The selector correction also separates solution mutation from output capture:
`git diff > patch.txt` is verification/finalization, not a new source mutation.
This prevents a redirected verification command from displacing the actual
`sed -i` edit in the mutation floor.  The focused and related regression suites
pass 201 tests.

Raw manifests, selection ledgers, logical metrics, and official results are
stored in the three subdirectories.  Engine/KV reuse is outside this bundle;
all token counts are logical ordinary-text materialization counts.
