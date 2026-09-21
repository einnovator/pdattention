# OpenHands Task 2 exact-tokenizer tail-90 qualification

Both arms use OpenHands SDK 1.49.2, `qwen3-coder:30b` at observed digest
`06c1097e...90bca`, its exact tokenizer revision, temperature zero, disabled
native condensation, the same task-derived image, and zero semantic-request
retries.  The selective arm uses matched token-tail with protected head 2,
protected tail 4, and a nominal 90% budget.

FULL resolves officially in 38 actions with 353,392 exact message-content
tokens.  Tail-90 also resolves officially, in 21 actions.  Its unmodified
trajectory contains 238,206 candidate tokens and materializes 215,196: 9.66%
own logical saving.  Against paired FULL, materialized input falls 39.11% and
provider prompt accounting falls 42.54%.  The candidate reaches 91.83% minimum
per-request logical retention and ends at 94.39%; mandatory records explain
the small declared-budget overflow and no overflow is unexplained.

The two valid patches differ.  FULL uses an `(Expression, F)` type check;
tail-90 uses the more general `resolve_expression` protocol check.  Both pass
the independent official grader, so patch identity is not used as a quality
criterion.  No request exceeds the 1,024-token completion ceiling (maximums
952 and 988).

The first FULL attempt for this task is separately quarantined because the
Ollama endpoint ignored `max_completion_tokens` on the wire.  These evidence
directories contain only the clean post-normalization pair; the transport
diagnostic is recorded in sibling
`ollama_completion_limit_wire_compatibility_v1` evidence.
