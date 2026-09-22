# Corrected OpenHands H2/T4 tail-90, Task 1

This directory records the first autonomous OpenHands execution after fixing
the matched-tail materializer to consume the declared protected-head and
protected-tail floors. The arm uses OpenHands 1.49.2, `qwen3-coder:30b` at
digest `06c1097e...90bca`, the exact matching tokenizer, temperature zero,
disabled native condensation, the locked SWE-bench image, and official
grading.

The corrected arm resolves `django__django-15277`. It produces the same
689-byte patch and patch digest as the paired successful FULL control, uses 31
actions rather than 33, materializes 287,641 of 444,913 tokens on its own
trajectory (35.35% own saving), and materializes 43.53% fewer tokens than the
predeclared FULL workload. Provider prompt accounting falls 33.27%.

This is a one-identity transfer result, not an accuracy estimate or production
default. It shows that the corrected floors can preserve official task quality
while providing materially more opportunity on a richer-agent history than on
the corrected three-task mini-swe-agent cohort. The difference is attributed
to workload structure, not engine behavior, pending repeats and Tasks 2--3.

`qualification.json` is the compact decision record. `proxy_trace.jsonl`
contains the exact logical selection ledger; engine K/V savings must later be
computed from selected original K/V identities rather than from rewritten or
re-encoded text.
