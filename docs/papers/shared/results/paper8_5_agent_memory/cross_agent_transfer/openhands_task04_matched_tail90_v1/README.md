# OpenHands Task 4 matched causal-tail qualification

This immutable artifact pairs the qualified OpenHands FULL control with a
prospective `matched_token_tail` run on `django__django-15741`. Both executions
use OpenHands SDK 1.49.2, `qwen3-coder:30b`, temperature zero, disabled native
condensation, pinned transport controls, and live execution receipts.

The selective run is officially resolved: 2/2 FAIL_TO_PASS and 104/104
PASS_TO_PASS tests pass. It finishes in 22 actions rather than FULL's 43.
Whole-record selection retires only 1.94% of logical source tokens on its own
trajectory. Ordinary-text boundary materialization reduces input by 9.33% by
slicing an oversized observation at the strict ceiling. Relative to FULL's
227,056 cumulative message-content tokens, the candidate materializes 62.51%
fewer tokens because it also reaches the solution in 21 fewer actions.

The logical and materialized figures are deliberately separate. Paper 4.5's
native-K/V transfer may claim the logical/KV-subset saving, but may not claim
the extra text-materialization saving unless its engine supports original-
position record-internal K/V spans without re-encoding.
