# Ollama completion-limit wire compatibility

This diagnostic was triggered by an OpenHands Task 2 FULL run that declared a
1,024-token completion ceiling but remained inside one generation for more
than seven minutes.  The Ollama server log showed more than 12,000 generated
tokens before cancellation.  The run was stopped and quarantined before it
could enter the paired policy result.

On the same locked `qwen3-coder:30b` endpoint and digest, a direct request with
`max_completion_tokens=4` was not bounded and timed out after 120 seconds.  An
otherwise identical request with `max_tokens=4` returned exactly four
completion tokens with `finish_reason=length` in 0.23 seconds.  The transfer
proxy now validates the logical limit and normalizes it to `max_tokens` on the
upstream wire.  A regression test asserts the forwarded payload, and the
OpenHands runner separately performs a Docker-network preflight before any
semantic request.

The invalid Task 2 attempt remains outside every accuracy, action-count and
token-saving aggregate.
