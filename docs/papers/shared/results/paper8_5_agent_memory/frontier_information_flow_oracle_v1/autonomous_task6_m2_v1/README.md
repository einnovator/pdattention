# Autonomous Task-6 M=2 pilot

This is the first autonomous execution of `frontier_dag_retirement`, forked
from the exact saved five-issue persistent-FULL prefix. It uses whole causal
group retirement, `qwen3-coder:30b`, temperature zero, and the last two user
prompts as the live frontier.

| Arm | Official result | Calls | Cumulative input tokens |
|---|---:|---:|---:|
| Paired historical FULL | 0/1 | 5 | 142,822 |
| Frontier DAG M=2 | 0/1 | 15 | 254,090 materialized |

The treatment removes 41.19% relative to its own 432,065-token full-history
counterfactual, but sends 77.91% more materialized input than the paired FULL
trajectory because it adds ten calls. This is a clear efficiency rejection for
this task execution. Since FULL also fails, it is not evidence of an accuracy
loss or preservation. The next quality test must use a paired FULL-success
task; Task 5 is the predeclared candidate.

The committed JSON files preserve the treatment metrics, official grader
result, manifest, and paired FULL metrics. The full request trace remains in
the immutable remote run directory recorded by the manifest.
