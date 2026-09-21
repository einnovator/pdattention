# OpenHands Task 3 exact-tokenizer tail-90 qualification

Both arms use OpenHands SDK 1.49.2, `qwen3-coder:30b` at observed digest
`06c1097e...90bca`, its exact tokenizer revision, temperature zero, disabled
native condensation, the same Scikit-learn task image, and zero semantic-request
retries.  The selective arm uses matched token-tail with protected head 2,
protected tail 4, and a nominal 90% budget.

FULL resolves officially in 20 actions with 252,532 exact message-content
tokens.  Tail-90 also resolves officially, in 19 actions.  Its unmodified
trajectory contains 250,916 candidate tokens and materializes 226,894: 9.57%
own logical saving.  Against paired FULL, materialized input falls 10.15% and
provider prompt accounting falls 7.21%.  This short task leaves little safely
retirable history: per-request logical retention never falls below 98.23% and
ends at 99.03%.

Both arms implement the same `np.sort(centers)` fix and pass the independent
official grader; only their explanatory comment differs.  No request exceeds
the 1,024-token completion ceiling (maximums 647 and 702).  This low-saving
pass is retained rather than hidden because task-clustered reporting must show
that the same frozen policy has strongly heterogeneous value by trajectory.
