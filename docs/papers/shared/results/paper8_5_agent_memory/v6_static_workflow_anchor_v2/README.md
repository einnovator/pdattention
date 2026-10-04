# Static workflow anchor v2 gate

This bundle tests a task-neutral workflow anchor encoded once in the immutable
system prefix.  It is not post-cache compaction: retired task-history K/V is
not summarized or re-encoded.  The selector is the boundary-free P0/W0 arm,
which keeps the active task state while retaining no contextualized workflow
or protocol exemplar from an earlier task.

## Frozen first-action gate

The same reconstructed request prefix, tokenizer, model endpoint, temperature
zero setting, and workspace evidence are used for FULL and the candidate.
Each arm is repeated twice.

| Task | FULL selected tokens | P0/W0 selected tokens | Reduction | FULL first action | P0/W0 first action | Repeat result |
|---|---:|---:|---:|---|---|---|
| `django__django-15741` (Task 4) | 22,073 | 1,521 | 93.1% | bounded `get_format` search | bounded `utils/formats` search | exact within arm, 2/2 |
| `django__django-14089` (Task 7) | 35,846 | 2,498 | 93.0% | bounded `OrderedSet` search | bounded `OrderedSet` search | exact within arm, 2/2 |

The Task-7 result is the important correction.  Anchor v1 with P0/W0 chose a
whole-file read.  Anchor v2 instead produces a bounded search in both repeats.
This proves the earlier failure was not an unavoidable consequence of retiring
old tasks; it was a missing task-neutral workflow prior.

These frozen rows qualify only the first action.  They do not establish task
resolution, total calls, or cumulative saving.  A matched autonomous FULL and
P0/W0 pair is required before this policy can be promoted to Paper 4.5.

## Autonomous Task-7 diagnosis

The initial autonomous pair exposed a decoding-control defect.  Although the
request declared temperature zero, the served model's Modelfile retained
`temperature=0.7`, `top_k=20`, and `top_p=0.8` defaults, and exact request
replays ranged from one valid action to twelve code blocks.  Explicit
`top_k=1` made the exact selective first request identical in 5/5 repeats.
Commit `52c1d3d1` therefore makes `top_k` an audited generation and pairing
coordinate; older temperature-zero results without it remain evidence for the
observed stochastic endpoint, not deterministic-trajectory controls.

The matched `top_k=1` pair used the same Task-7 prefix, model, tokenizer,
static anchor, workspace image, and official grader:

| Arm | Official | Calls | Cumulative selected input | Own saving | Paired saving vs FULL |
|---|---:|---:|---:|---:|---:|
| Anchored FULL | 1/1 | 6 | 213,287 | 0.0% | 0.0% |
| Anchored P0/W0 | 1/1 | 14 | 52,807 | 89.8% | 75.2% |

This is an accuracy-positive but workflow-cost-negative point.  Closed-task
retirement preserved the official solution and very large cumulative input
saving, but added eight tool calls.  The added work is localized: filename
search before content search, a pre-edit behavior test, fragile `sed` insertion,
and repeated inspect/revert/re-edit cycles.  All active-task messages were
retained, so this is not loss of the current decision spine.  It is loss of
workflow conditioning combined with a weak Bash-only editing affordance.

A single structural counterfactual retained the immediately preceding task
whole (`R2/P0/W0`).  It restored a targeted first search but was stopped after
15 recorded selections (16th request in flight): 171,990 selected of 567,600
counterfactual FULL tokens, 69.7% own saving, no terminal result.  It repeated
the same inspect/revert/`sed` loop.  Therefore the next intervention is not a
larger retention sweep.  It is a task-neutral, pre-ingest reliable-edit
affordance or typed edit primitive, paired identically in FULL and selective
arms.  No post-cache summary or re-encoding is permitted.

## Provenance

- Source branch: `research/paper8-5-agent-memory`
- Frozen-run source revision: `2b1eba46`
- Greedy autonomous source revision: `52c1d3d1`
- Anchor: `coding_bounded_evidence_v2`
- Served model: `qwen3-coder:30b-ctx131k`
- Greedy temperature/top-p/top-k/seed: `0.0 / 1.0 / 1 / 0`
- Frozen rows: `2 FULL + 2 candidate` per task
- Regression suite: `93 passed` across the autonomous proxy and frozen-oracle
  tests on the execution host.
