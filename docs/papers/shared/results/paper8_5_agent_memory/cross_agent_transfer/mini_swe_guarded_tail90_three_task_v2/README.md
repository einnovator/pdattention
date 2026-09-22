# Guarded mini-swe-agent three-task tail-90 cohort

> **Post-refactor audit:** these immutable totals reproduce exactly, but the
> historical materializer did not consume the recorded head/tail floors. This
> bundle is evidence for the prompt-pinned pure-recency policy that actually
> ran, not the corrected H2/T4 policy. Fresh corrected runs supersede it for
> policy promotion.

This cohort uses mini-swe-agent 2.4.6 and the same
`qwen3-coder:30b` endpoint, model digest, tokenizer, temperature-zero settings,
workspace images, and structural unified-diff submission guard in every arm.
Each task has two independent FULL controls followed by the legacy matched
token-tail implementation at a nominal 90% budget ceiling.

| Task | FULL-A solve/calls/tokens | FULL-B solve/calls/tokens | tail-90 solve/calls | own saving | paired saving vs FULL-A |
|---|---|---|---|---:|---:|
| `django__django-15277` | 1 / 24 / 112,566 | 1 / 27 / 164,726 | 1 / 27 | 12.93% | -25.95% |
| `django__django-15368` | 1 / 15 / 60,357 | 1 / 21 / 86,265 | 1 / 16 | 13.10% | 0.24% |
| `scikit-learn__scikit-learn-13135` | 1 / 23 / 199,109 | 1 / 21 / 188,652 | 1 / 19 | 9.67% | 32.13% |

Across the three task clusters, official resolution is 3/3 for FULL-A, 3/3
for FULL-B, and 3/3 for tail-90.  FULL-A and tail-90 both require 62 aggregate
calls.  Tail-90 materializes 337,133 of 381,737 candidate-own full-history
tokens, an 11.68% own saving.  Relative to the predeclared FULL-A controls it
saves 9.38% cumulative input tokens with no aggregate call increase.

FULL-repeat call counts vary materially even at temperature zero (24/27,
15/21, and 23/21).  Task 1's selective trajectory diverges before its first
selection change, so the negative paired value there cannot be assigned
cleanly to selection.  The paper must retain both the strict predeclared pair
and this repeat-instability qualification.

This is a quality-preserving bridge/control, not the target policy: it remains
well below the 30--50% logical-saving objective.  `tradeoff_spec.json` rebuilds
the task-clustered curves from the three immutable evidence directories.
