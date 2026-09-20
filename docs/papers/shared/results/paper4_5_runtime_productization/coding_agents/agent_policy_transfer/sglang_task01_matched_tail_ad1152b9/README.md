# SGLang-MLX matched-tail transfer: Task 1

This immutable cell transfers Paper 8.5's frozen
`paper8.5-matched-causal-token-tail-v1` policy to the qualified SGLang-MLX
native-K/V path on `django__django-15277`.

The autonomous run is a valid policy-quality negative. The official SWE-bench
grader reports `0/1` after 9 calls. Aggregate realized retention was 89.57%,
selected-history re-encoding was zero tokens, selected-K/V physical copy was
zero bytes, and host-to-device transfer was zero bytes. Suffix grafting copied
161,808,384 device bytes and consumer temporaries peaked at 1,610,612,740
bytes; these are reported separately and are not described as selected-K/V
reuse.

The first action divergence from the qualified plain/PRA-100 trajectory occurs
at request 3, immediately after the failed `nano` action. The frozen policy
retains the task and the current failed-action/recovery bundle but retires the
older `grep` discovery bundle (223 source tokens, 9.31% of live history). The
model then chooses an invalid multi-line `sed` command rather than the simple
one-line replacement used by both plain and PRA-100. Repeated edit recovery
eventually empties the target file, producing a 94,017-byte, 2,544-line patch.
Because request 3 has exactly one omitted causal bundle, the paired plain and
PRA-100 controls are also the one-bundle oracle add-back reference. This
localizes the failure to policy/model transfer rather than history
re-encoding, K/V copying, or a chat-template mismatch.

The earlier `run1` directory is excluded: Docker container startup timed out
before the first model call. This directory contains only the clean immutable
retry (`run2`).
