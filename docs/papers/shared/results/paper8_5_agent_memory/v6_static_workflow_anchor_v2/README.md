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

## Provenance

- Source branch: `research/paper8-5-agent-memory`
- Frozen-run source revision: `2b1eba46`
- Anchor: `coding_bounded_evidence_v2`
- Served model: `qwen3-coder:30b-ctx131k`
- Temperature/top-p/seed: `0.0 / 1.0 / 0`
- Frozen rows: `2 FULL + 2 candidate` per task
- Regression suite: `93 passed` across the autonomous proxy and frozen-oracle
  tests on the execution host.
