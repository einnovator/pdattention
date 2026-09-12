# Autonomous Task 02: replicated FULL-control instability

This directory contains two fresh-workspace FULL-history controls for
`django__django-15368`. Both use mini-swe-agent 2.4.6, `qwen3-coder:30b`, its
pinned tokenizer, temperature 0, top-p 1, seed 0, a 1,024-token completion
ceiling, whole-record text, and the official SWE-bench Docker grader.

| Control | Primary official submission | Auxiliary final workspace | Calls | Full/selected tokens | Exact pass-through | Reacq. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL-A | 1/1 | unavailable | 30 | 153,318 / 153,318 | 30/30 | 0 |
| FULL-B | 0/1 | 1/1 | 13 | 42,951 / 42,951 | 13/13 | 0 |

FULL-B made a source change but submitted a source excerpt rather than a
unified diff, so its primary official grade is `patch_apply_failed`. A separate
post-hoc official grade of the provenance-checked 731-byte terminal-workspace
patch (`f5d867f28cde646c6c844a3ea9745075b0ab696268f0c7f79508e37f20922158`)
resolves 1/1. FULL-B is therefore a submission-protocol failure rather than a
solution-state failure. The auxiliary outcome does not replace or relabel the
primary submission endpoint.

FULL-A's primary submission resolves. Auxiliary extraction is unavailable
because its terminal checkpoint contains the untracked files
`bulk_update_fix.patch` and `test_fix.py`; the extractor fails closed rather
than grading an incomplete tracked-only representation.

The controls started from the same image and exact initial workspace and
environment fingerprints. Requests are identical through call 5. The model's
reasoning first differs at call 5 despite consuming the same request, while the
command remains equal; request history and commands first differ at call 6.
Thus temperature zero and a fixed seed are recorded generation settings, not a
claim of deterministic autonomous behavior.

This independently repeats Task 01's primary 1/2 FULL submission outcome, but
the auxiliary grade shows that both FULL-B runs reached resolving code states.
The remaining instability is in calls, actions, and submission-protocol
adherence, not evidence that either FULL-B failed to find a solution. The
predeclared Task 02 gate required both primary FULL submissions to resolve with
sufficiently stable behavior. It failed, so no Task 02 H1, H2a, H3, or combined
heuristic execution was run. Strict negative-selection rules remain fail-closed
rather than being relaxed to manufacture a policy row.

`comparison.json` is the compact reduction. Each control directory contains
the run manifest, cumulative metrics, per-request trace, full trajectory, and
primary normalized/raw grader result. The `auxiliary_workspace_state*`
artifacts separately record extraction availability, patch identity, validation
checks, and auxiliary grading.
