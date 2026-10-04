# V5 first-divergence oracle: removing stale exemplars

This bundle is a frozen-prefix diagnostic, not autonomous task-quality
evidence.  It replays the exact model-visible request at the first important
divergence of two persistent-session failures from
`ATOMIC_FRONTIER_M1_P1_W1_U1`.  The consumer is the same
`qwen3-coder:30b-ctx131k` model (`4abd6222...`), with temperature 0, top-p 1
and seed 0.  The diagnostic completion ceiling is 256 tokens; therefore only
the validity and semantics of the next action are interpreted.

The counterfactual changes only the number of retained completed protocol and
workflow exemplars from `P1/W1` to `P0/W0`.  Current instructions, the active
causal turn, one recent completed turn, mutation/verification state and the
superseded-component rule remain unchanged.

## Frozen next-action results

| Task / request | Arm | Selected / FULL tokens | Saving | Repeats | Next action |
|---|---:|---:|---:|---:|---|
| Task 4 / request 1 | FULL | 21,972 / 21,972 | 0% | 2/2 | task-correct `get_format` search |
| Task 4 / request 1 | M1/P1/W1/U1 | 7,425 / 21,972 | 66.21% | 2/2 | resumes the prior `KBinsDiscretizer` task; no valid action before the 256-token ceiling |
| Task 4 / request 1 | M1/P0/W0/U1 | 1,420 / 21,972 | 93.54% | 2/2 | task-correct `utils/formats` search, exactly repeated |
| Task 7 / request 2 | FULL | 34,872 / 34,872 | 0% | 2/2 | targeted `grep -A 50` for `OrderedSet` |
| Task 7 / request 2 | M1/P1/W1/U1 | 3,922 / 34,872 | 88.75% | 2/2 | whole-file `cat` |
| Task 7 / request 2 | M1/P0/W0/U1 | 1,524 / 34,872 | 95.63% | 2/2 | whole-file `cat` |

Task 4 identifies a concrete selector error.  P1/W1 retained the immediately
preceding successful task as a workflow exemplar.  The model explicitly
reported conflicting Django and `KBinsDiscretizer` instructions and resumed
the stale task.  Removing the exemplar restores the correct next action in
both repeats.  Adding back either older completed epoch also produces a
task-correct action, so the failure is not caused by insufficient old history.

Task 7 is different.  Removing P1/W1 does not restore FULL's targeted read.
Adding back *any* one of the five retired completed epochs changes the action
to a targeted `grep`; the smallest such epoch adds 3,424 tokens.  Because
unrelated epochs all have this effect, the evidence does not establish a
task-specific causal dependency.  It is consistent with generic workflow
conditioning or a context-shape effect.  Autonomous success and call count,
not exact command equality, are the required gate.

## Harness corrections

The replay uncovered and fixed two provenance bugs before these results were
accepted:

1. commit `3f245ea3` locates the frozen request by its immutable pre-selection
   input digest, then independently validates the selected-history digest;
2. commit `05b70d9d` reconstructs an earlier decision from the exact leading
   receipt prefix of a post-campaign instrumentation archive, while retaining
   strict step and command-digest checks.

The next gate is an autonomous frozen-prefix run of Tasks 4 and 7 under
M1/P0/W0/U1.  P0/W0 is not a deployable default until it preserves official
resolution and does not increase aggregate calls.

