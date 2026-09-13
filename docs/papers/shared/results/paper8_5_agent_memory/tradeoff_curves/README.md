# Paper 8.5 adaptive trade-off curves

This directory is the first common reduction of Paper 8.5 selection evidence.
It keeps two evidence classes separate:

- autonomous official task resolution is the primary quality endpoint;
- frozen conservative next-action agreement is a screening and pruning proxy.

The current reduction contains 27 repeat-qualified frozen observations from
six matched cohorts and five autonomous executions. One frozen cohort is an
explicit same-materialization threshold repeat and is excluded from adaptive
promotion counts. The autonomous curve is
intentionally sparse: four FULL runs resolve 2/4 primary submissions, while
the only qualified heuristic execution, H3 on `django__django-15277`, resolves
0/1. This is not yet a population quality--saving curve.

The two declared FULL-repeat controls also expose the stochastic envelope:
their first tool-command divergences occur at calls 9 and 6, despite identical
temperature-zero decoding settings, and both second executions fail after the
first executions solve. H3 changes selected history at call 6 and first
diverges at call 7. Because neither H3 nor its paired FULL control resolves,
the currently observed 44.5% paired token reduction and -10 tool calls are
failure-termination diagnostics, not efficiency. The successful-pair panel is
therefore intentionally empty.

## Metric contract

- **Gross saving** is `1 - materialized/full` over the calls on the candidate's
  own trajectory.
- **Paired net saving** is `1 - candidate materialized tokens / paired FULL
  materialized tokens`; it includes differences in the number of calls.
- **Tool-call delta, all** includes failed and successful candidate runs.
- **Tool-call delta, successful** includes only pairs in which both the
  candidate and its FULL control resolve officially. A shorter failed run is
  never presented as an efficiency gain.
- **First action divergence** compares tool-command hashes against a declared
  paired FULL execution and is reported separately for every task.
- **Accuracy** means official SWE-bench resolution. Auxiliary final-workspace
  resolution remains diagnostic and never replaces the primary endpoint.

## Adaptive decision rule

The machine-readable decisions in `adaptive_decisions.csv` prevent a blind
Cartesian expansion:

- immediate/delayed H1, H2a, H1+H3, H2a+H3, and the all-guarded whole-group
  combinations stop because proxy agreement falls too quickly for their small
  saving;
- H4 remains an aggressive mechanism bound, not a profile candidate;
- certified duplicate exclusion and H3 remain mechanism controls;
- exact matched token tail and positional materializers remain controls and
  are not parents for strategy combinations;
- structured evidence receives one bounded autonomous probe because it keeps
  causal groups and removes only oversized observation detail. It does not
  expand to more tasks unless official quality passes;
- H2b's zero-saving result is an abstention control, not a promoted policy.

The thresholds and policy-family roles are predeclared in
`experiments/paper8_5_agent_memory/configs/tradeoff_current.json`.

## Outputs

- `tradeoff_curves.json`: complete normalized evidence and metric definitions;
- `frozen_observations.csv`: proxy observations with source hashes;
- `autonomous_observations.csv`: primary/auxiliary outcomes, calls and tokens;
- `autonomous_strategy_summary.csv`: task-macro quality, ratio-of-sums workload
  saving, and all-run versus doubly-successful paired deltas;
- `adaptive_decisions.csv`: stop/control/probe/promotion decisions;
- `frozen_saving_vs_action_agreement.{pdf,png}`;
- `autonomous_saving_vs_accuracy.{pdf,png}`;
- `autonomous_net_saving_vs_accuracy.{pdf,png}`;
- `autonomous_gross_saving_vs_tool_call_delta.{pdf,png}`;
- `autonomous_net_saving_vs_tool_call_delta.{pdf,png}`;
- `autonomous_saving_vs_first_action_divergence.{pdf,png}`.

The next admitted observation is a FULL Qwen3-14B autonomous control on the
cached scikit-learn task. Structured evidence is run on that pair only if the
FULL arm resolves. The 30B campaign resumes when the Big Mac is no longer
under CPU and disk pressure.
