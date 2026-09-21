# mini-swe-agent Task 1 FULL-repeat divergence

Two unselected FULL controls used the same locked Task 1 workspace,
`qwen3-coder:30b` digest, tokenizer, temperature zero, top-p one, seed zero,
scaffold and mini-swe-agent 2.4.6 harness. Run A officially resolved in 27
actions. Run B reached the same minimal workspace edit in 17 actions, but its
terminal command emitted source context rather than a Git patch and the
official grader correctly classified the primary submission as
`patch_apply_failed`.

The first 14 model-visible messages, covering six complete actions and
observations, are byte-identical. The first changed assistant action is action
7: the controls differ only by `grep` versus `grep -n` at that point. Therefore
the repeat fork precedes submission and is backend/model trajectory variability
despite temperature zero; it is not caused by PRA selection.

`divergence_audit.json` binds both immutable raw trajectories by SHA-256.
The raw runs remain under the campaign output root on the execution host. The
subsequent v2 campaign uses an identical, task-agnostic submission guard in
FULL and selective arms. It returns malformed terminal payloads as recoverable
nonzero observations and never substitutes a workspace patch for the primary
submission.
