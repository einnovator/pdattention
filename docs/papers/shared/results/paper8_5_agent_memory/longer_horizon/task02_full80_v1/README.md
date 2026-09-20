# Longer-horizon FULL control: Task 2

This immutable retry is the second task in
`easy50_horizon_limit5.json`: `django__django-16899`. It uses mini-swe-agent
2.4.6, `qwen3-coder:30b`, FULL history, temperature 0, top-p 1, seed 0, a
1,024-token completion ceiling, and an 80-call action ceiling.

The agent stops normally after 22 model calls and 21 actions, so the larger
horizon is not exercised. It submits a 1,067-byte source patch replacing the
readonly-field error with `refer_to_missing_field()`. The patch applies and all
56 PASS_TO_PASS tests pass, but both FAIL_TO_PASS tests fail because the helper
emits the wrong message contract. The official outcome is unresolved.

The run consumes 146,335 cumulative logical input tokens and 3,078 reported
completion tokens in 466.0 seconds. It includes 11 classified reads, three
searches, and one test; eight reads repeat an operation/resource signature.
The final workspace receipt is unavailable because the submission command is
not certified read-only, but the primary submitted patch and common grader are
complete.

This is a capability failure rather than a horizon-limit failure. It has no
passing target test and therefore fails the predeclared forward-progress gate;
it does not advance to 120 calls.

