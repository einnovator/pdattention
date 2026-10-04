# V5 autonomous closed-task retirement: R3/P0/W0

This bundle tests the policy implied by the first-divergence oracle: retire all
closed prior-task components, retain the active task, and do not pin a prior
task as a protocol or workflow exemplar.  The runs use the same frozen
persistent prefixes, Qwen3-Coder-30B endpoint, tokenizer, temperature 0,
top-p 1 and seed 0 as the contemporaneous FULL controls.

`R3` is retained in the strategy label because the configured recent-turn
floor is three.  The trace audit shows that it is not the operative limit:
the live task component remains fully selected.  At Task 7 request 40 all 80
active-task messages are selected; the only exclusions are six closed prior
components.

## Autonomous outcomes

| Task | FULL official / calls / tokens | R3/P0/W0 official / calls | Candidate selected / own-FULL tokens | Own saving | Raw paired saving | Failure-aware result |
|---|---|---|---:|---:|---:|---|
| Task 4 (`django__django-15741`) | pass / 13 / 300,312 | pass / 12 | 96,069 / 342,693 | 71.97% | 68.01% | pass; one fewer call |
| Task 7 (`django__django-14089`) | pass / 10 / 357,028 | fail / 40 | 318,996 / 1,652,916 | 80.70% | 10.65% | zero saving credit; +30 calls |

Task 4 confirms the stale-exemplar diagnosis.  The rejected P1/W1 run resumed
the preceding `KBinsDiscretizer` task and failed in six requests.  P0/W0
instead resolves officially in 12 requests.  A separately stopped R1/P0/W0
execution crossed the FULL call count and produced a duplicated malformed
edit; it is a dominated trajectory diagnostic, not an accuracy observation.

Task 7 falsifies the hypothesis that a deeper recent-turn floor alone fixes
the policy.  FULL follows a disciplined 10-action sequence: search, targeted
read, bounded edit, inspect, revert, precise edit, verify and submit.  With
closed tasks removed, the agent starts with `ls`, reads the whole file, applies
a global `sed` insertion to every `__len__` method, and spends the remainder of
the 40-action horizon repairing its own corruption.  No active-task message
was retired.  Thus the excess calls are caused by loss of cross-task workflow
conditioning, not loss of the current task's progress state.

The next policy must preserve workflow without preserving stale task
semantics.  The predeclared candidate is an **action-skeleton exemplar**:
selected command/control spans and success/error status from completed
workflows, but no prior task statement, free-form rationale, resource payload,
or file-specific result content.  This is a region-selection policy over
existing records/K/V, not post-cache textual compaction.  It must first repair
Task 7 without breaking Task 4 before expansion to the remaining Easy-14
cohort.

