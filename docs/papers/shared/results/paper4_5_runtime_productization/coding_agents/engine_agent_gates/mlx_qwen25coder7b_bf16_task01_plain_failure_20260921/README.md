# MLX Qwen2.5-Coder-7B BF16 Task 1 plain admission

This immutable directory preserves the first strict-checkpoint plain admission
attempt for `django__django-15277`. The model used the pinned BF16 source
revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, mini-swe-agent 2.4.6,
temperature zero, and a 30-step limit.

The agent reached `LimitsExceeded` with an empty patch. This is a plain
model/task admission failure, not a PRA failure, so no selective arm is
interpretable for this task. The official grader subsequently encountered a
Docker cleanup race while reporting the empty-patch execution; the empty patch
and `LimitsExceeded` status are already fixed by the agent artifacts.

The server in this diagnostic predated the fail-closed observed-revision health
receipt added in commit `9eef7249`; the local snapshot path itself was the exact
pinned revision. Task 5 is the next strict, receipt-qualified admission cell.
