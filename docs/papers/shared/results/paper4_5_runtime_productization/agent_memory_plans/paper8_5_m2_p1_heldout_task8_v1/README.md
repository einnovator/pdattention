# Held-out M2/P1 Task 8 native-K/V handoff

This directory is the immutable Paper 4.5 handoff for held-out identity
`django__django-15277`, episode 8 of the completed Paper 8.5 baseline-conditional
campaign.  The exact-prefix FULL control and frozen M2/P1 treatment both resolve
officially in six calls.  Along the paired trajectories, message-content input
falls from 236,111 to 89,534 tokens (62.08%).  Those are Paper 8.5 logical-policy
results, not native-engine savings.

The export was produced by Paper 8.5 commit `b5e1969f` from campaign-state
SHA-256 `7c4a47f5c06a966710ec69517fb625054395b692855d221a0ad5049035b22c7a`.
It reconstructs and hash-validates all six original requests; it does not rerun
the selector.  The source prefix contains seven completed issues, making this a
direct test of cross-task resident-K/V retirement rather than a short synthetic
history.

Files:

- `frozen_plan.jsonl`: selected logical records translated to stable resource
  spans;
- `request_replay.jsonl`: exact full logical requests and recorded responses;
- `frozen_plan_manifest.json`: source and output digests;
- `qwen3coder30b_tokenizer_audit.json`: exact Qwen3-Coder tokenizer geometry.

The tokenizer audit realizes 36.46%--38.22% retention across the six requests,
equivalent to 61.78%--63.54% selected-history token saving, without restoring an
excluded record.  Paper 4.5 must now consume these exact fixtures through each
engine's native resident-K/V path and independently report selected-history
re-encoding, physical K/V copy, consumer temporary bytes, position handling,
and lifecycle qualification.  A tokenizer audit alone is not native-K/V
execution evidence.
