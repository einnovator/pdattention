# Qwen2.5-Coder-14B tool-order control

This bundle isolates a non-PRA cause of autonomous trajectory divergence on
`django__django-15277`.  All runs use FULL history, mini-swe-agent 2.4.6,
Qwen2.5-Coder-14B-Instruct-4bit revision `29efdbab55a1`, temperature zero,
top-p one, seed zero, the backtick scaffold, a unified-diff submission guard,
and official SWE-bench grading.

Two fresh post-refactor controls are byte-identical across all seven assistant
responses.  They materialize 17,410 message-content tokens, submit the same
945-byte malformed patch, and grade 0/1.  Their complete assistant-trajectory
digest is `bd73f6c...e2c3`; their patch digest is `a6c6e0a3...f3b84`.  These
exactly reproduce the earlier same-endpoint FULL failure, so the common
record/DAG refactor did not introduce the regression.

A cross-branch comparison with Paper 4.5's successful direct-MLX control found
the first causal difference.  System text, task text, first assistant response,
first command, return code, and the six search matches are identical.  Only
the line order of the recursive `grep` observation differs.  The successful
run begins with test paths; the failed `.8` run begins with the target Django
path.  The second assistant action then diverges.  Thus temperature-zero does
not make an agent trajectory deterministic when a set-valued tool observation
has nondeterministic enumeration order.

Revision `b7fc1866` adds an opt-in evaluation control that lexicographically
orders independent lines from simple, context-free `find` and recursive
`grep` commands.  It rejects pipelines, redirects, context flags, NUL output,
and JSON output, and records original and visible observation hashes.  The
canonicalized FULL run also grades 0/1 in seven calls and produces the same
malformed patch.  Canonicalization therefore removes this variance source but
does not turn Task 1 into a favorable model/task admission.  No selective arm
is qualified against these failed FULL controls.

A paired frozen-prefix replay then removes the remaining environment and
trajectory variables.  On the same current endpoint, the successful ordering
reproduces its recorded second assistant action byte for byte, and the failed
ordering reproduces its different recorded second action byte for byte.  Both
requests have the same system/task/first-action hashes and differ only in the
first observation hash.  This establishes that observation order alone is
sufficient for the immediate action fork; it does not establish that order
alone determines the final task outcome.  Receipts are under `prefix_replay/`.

The run directories contain the run manifest, request ledger, logical
metrics, and official result.  `audit.json` records message and patch hashes
and the exact first-observation comparison.  The canonicalization is an
evaluation/tool-semantics control, not a PRA policy and not a token-saving
claim.
