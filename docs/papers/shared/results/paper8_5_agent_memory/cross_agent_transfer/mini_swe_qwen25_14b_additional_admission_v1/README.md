# Additional Qwen2.5-Coder-14B mini-swe-agent admissions

These FULL-history diagnostics use the same Qwen2.5-Coder-14B revision and
canonical unordered-search observation control as the Task 1 tool-order audit.
No PRA selection occurs.

Task 4 (`django__django-15741`) reaches the declared 20-call limit after 19
executed actions. It mutates the workspace repeatedly but never submits a
patch; official grading is 0/1. The run materializes 100,638 cumulative
message-content tokens. This is a plain model/scaffold failure, so no selective
arm is launched.

Task 5 (`pytest-dev__pytest-7982`) is a stopped no-progress diagnostic, not an
official benchmark result. After 16 model calls and 16 executed actions, every
captured pre/post workspace fingerprint is unchanged. The run had already
materialized 102,919 cumulative message-content tokens and repeated a short
set of actions. It was stopped rather than spending the full 40-call ceiling.
The request ledger and all execution receipts are retained; no solve-rate or
policy claim uses this row.

Together with the Task 1 controls, these diagnostics show that the 14B `.8`
endpoint currently lacks a repeat-qualified plain-success identity for the
planned cross-agent policy pair. That is an admission limitation, not evidence
against PRA selection.
