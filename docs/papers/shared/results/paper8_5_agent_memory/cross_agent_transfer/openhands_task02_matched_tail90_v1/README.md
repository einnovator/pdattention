# OpenHands Task 2 matched causal-tail qualification

This immutable artifact pairs the qualified OpenHands FULL control with a
prospective `matched_token_tail` run on `django__django-15368`. Both executions
use OpenHands SDK 1.49.2, `qwen3-coder:30b`, temperature zero, disabled native
condensation, pinned 1,200-second request timeout, zero request retries, and
complete live execution-receipt delivery.

The selective run is officially resolved: 1/1 FAIL_TO_PASS and 29/29
PASS_TO_PASS tests pass. It finishes in 33 actions rather than FULL's 38.
Across its own trajectory it exposes 148,606 full-history tokens and
materializes 132,438, a 10.88% own saving. Relative to FULL's 153,174
cumulative message-content tokens, it materializes 13.54% fewer tokens. As in
Task 1, own saving and paired saving are reported separately because the
selective trajectory is shorter.

The nominal 90% setting is a strict materialized-token ceiling. Realized
request-level retention ranges from 82.18% to 100%; whole-turn packing leaves
2,028 aggregate budget tokens unused. `proxy_trace.jsonl` is the per-request
selection ledger, while the native events and official reports preserve the
agent and task-quality evidence.
