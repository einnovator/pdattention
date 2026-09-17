# Paper 8.5 E2+F1C engine handoff

This directory is the immutable Paper 4.5 handoff for the successful
`scikit-learn__scikit-learn-14496` E2+F1C treatment from Paper 8.5 commit
`3aa213b6`.

The nine-request plan was exported without rerunning the selector. The exporter
reconstructed each FULL logical request from the frozen five-episode prefix and
current trajectory, validated the recorded request hashes, then applied the
recorded wire-plan replacements for compact closure receipts. Therefore this
fixture represents the actual model-visible treatment, not an approximation
that omits materialization.

- `frozen_plan.jsonl`: ordered selected resources and original logical record
  positions, segmented into at most 256-token physical resources;
- `request_replay.jsonl`: exact FULL logical requests and frozen generation
  settings for identical-subset engine references;
- `frozen_plan_manifest.json`: source and output hashes.

Logical task success and the 38.53% own-trajectory saving remain Paper 8.5
claims. Paper 4.5 must independently report same-subset exactness, selected
history re-encoding, K/V-copy bytes, consumer temporaries, and lifecycle
correctness for each engine. No request-level engine smoke is an autonomous
task-accuracy result.
