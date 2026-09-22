# Corrected mini-swe-agent H2/T4 tail-90 cohort

This is the first autonomous cohort after fixing the matched-tail materializer
to consume its configured protected head and tail floors. Every request keeps
all genuine prompts, the first two complete causal turns, and the latest four
complete causal turns; only the intervening whole turns are selected to meet a
nominal 90% materialized-token ceiling. The runs use mini-swe-agent 2.4.6,
`qwen3-coder:30b` at digest `06c1097e...90bca`, its exact tokenizer,
temperature zero, the locked SWE-bench images, and official grading.

| Task | FULL-A/FULL-B/candidate solved | Calls A/B/C | Own saving | Paired saving vs A |
|---|---:|---:|---:|---:|
| `django__django-15277` | 1 / 1 / 1 | 23 / 27 / 20 | 9.35% | 22.88% |
| `django__django-15368` | 1 / 1 / 1 | 15 / 21 / 22 | 8.11% | -48.93% |
| `scikit-learn__scikit-learn-13135` | 1 / 1 / 1 | 23 / 21 / 22 | 7.62% | 14.27% |
| **Pooled workload** | **3 / 3 / 3** | **61 / 69 / 64** | **8.17%** | **6.28%** |

The candidate preserves every paired official success but fails the
predeclared efficiency gate against FULL-A: it adds three aggregate calls, and
the task-macro paired saving is -3.93%. Against FULL-B it uses five fewer calls
and saves 22.31% pooled input, demonstrating why both the predeclared pairing
and repeat-control envelope must be retained.

Task 1 has a second corrected selective repeat. It also resolves officially,
with 26 calls and 11.28% own saving. In that repeat the first action divergence
from FULL-A occurs at action 2, while the first selection change is request 8;
the early fork is therefore backend/model trajectory variability. The two
corrected Task-1 candidates jointly establish 2/2 quality preservation but not
stable calls-to-solution.

The four Task-2/Task-3 FULL controls reproduce the historical pre-refactor
controls exactly at the official outcome, call, and cumulative-token levels:
15/60,357; 21/86,265; 23/199,109; and 21/188,652. Immutable re-reduction also
reproduces the historical legacy-tail totals. The refactor did not change the
baseline or reducer; the corrected H2/T4 policy differs because the old
materializer had ignored those floors.

The archived legacy mini-swe and OpenHands rows remain evidence for the actual
prompt-pinned pure-recency policy that ran. They must not be labeled H2/T4 or
used to predict the corrected policy's savings.
