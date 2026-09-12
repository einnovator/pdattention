# Paper 8.5 pilot structural screen

This directory begins the engine-independent agent-memory study. The pilot
replays the message geometry of three successful mini-swe-agent trajectories:

- `django__django-15277` (23 model decisions);
- `django__django-15368` (30 model decisions);
- `scikit-learn__scikit-learn-13135` (16 model decisions).

At each of the 69 historical decision points, the screen recordizes the
available history and evaluates 700 combinations: five budget fractions, four
head floors, five tail floors, five middle policies, and two DAG-first
compositions. This produces 48,300 policy decisions, aggregated into 2,100
summary rows in
`pilot_structural_screen.json`.

The token counter is `whitespace_v1_structural_only`. Consequently, this
artifact establishes only policy geometry, bundle round-up, and mandatory
overflow. It is **not** evidence of next-action agreement, autonomous task
quality, model-token savings, K/V reuse, or runtime improvement. Model-facing
experiments must use the exact frozen tokenizer and exact workspace snapshots.

For the illustrative `(head=1, tail=2)` slice, aggregate realized retention
across all three trajectories is:

| Policy | 90% request | 75% request | 50% request | 25% request |
|---|---:|---:|---:|---:|
| Middle none | 55.5% | 55.5% | 55.5% | 55.5% |
| Progress spine | 66.5% | 66.5% | 66.5% | 66.5% |
| Middle recency | 85.9% | 76.7% | 59.6% | 55.5% |
| Middle lexical | 89.5% | 77.1% | 59.6% | 55.5% |
| Spine + lexical | 91.6% | 80.0% | 67.4% | 66.5% |

The key structural result is that percentages are ceilings, not guaranteed
realized retentions. Whole causal bundles produce unused capacity, while the
mandatory task/head/tail/spine can round above the requested budget. At 90%,
the mandatory `(head=1, tail=2)` base exceeds the nominal ceiling at 16 of 69
decision points; adding the role-aware spine raises this to 22 of 69. These
overflows occur mainly while the trajectory is short and immutable state is a
large fraction of the history. They must be reported, not hidden by splitting
causal turns.

The legacy trajectories do not record complete cwd/environment/resource-version
identity. The `DAG_CERTIFIED` arm therefore fails closed: at `B=100%` it retains
100% of logical history. Across the 69 growing decision prefixes, static Bash
analysis reports 458 superseded-current-state candidate occurrences and 75
write-invalidated-state occurrences, but no trace-exact duplicate and no
certified exclusion. These counts are repeated opportunities across prefixes,
not unique records, and none is removed. This is an instrumentation finding,
not a task-quality result.

Source trajectories are read-only inputs from the Paper 4.5 pass-through
campaign. Their successful completion only qualifies them as reference
histories; it does not validate any policy in this directory.
