# Guarded mini-swe-agent Task 1 FULL--tail-90 evidence

This bundle is the first run under the structural unified-diff submission
guard.  All three arms resolve `django__django-15277` with the official
SWE-bench grader using mini-swe-agent 2.4.6, `qwen3-coder:30b` digest
`06c1097e...90bca`, temperature zero, and the same immutable workspace image.

| Arm | Official solve | Calls | Full logical tokens | Materialized logical tokens | Own saving |
|---|---:|---:|---:|---:|---:|
| FULL-A | 1 | 24 | 112,566 | 112,566 | 0% |
| FULL-B | 1 | 27 | 164,726 | 164,726 | 0% |
| matched tail-90 | 1 | 27 | 162,842 | 141,779 | 12.93% |

The predeclared FULL-A pairing gives `-25.95%` paired input-token saving
because the selective trajectory takes three extra calls.  That number is not
a clean selection effect: FULL-B also takes 27 calls, and both FULL-B and
tail-90 diverge from FULL-A before selection can act.  In the tail-90 pair the
first assistant-action divergence is action 2, while the first selection
change is request 3.  Relative to the equal-call FULL-B repeat envelope,
tail-90 materializes 13.93% fewer input tokens; this is a diagnostic envelope,
not a replacement for the predeclared pair.

The result therefore establishes task preservation and measurable logical
saving, but not deterministic causal attribution of call count.  The complete
schema-2 pairing identities and reducer outputs are in `curves/`.  Large
instrumentation snapshots and Docker grader directories are intentionally not
duplicated in this repository bundle; their hashes and source paths remain in
the manifests and metrics.

Code revisions:

- execution: `86b36ab5880e8dd76508af9b98ab77199a759bd9`;
- strict observed-platform reducer: `7f804f6f`.
