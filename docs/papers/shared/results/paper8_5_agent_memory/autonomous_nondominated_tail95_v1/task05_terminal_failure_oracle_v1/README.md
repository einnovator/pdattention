# Task 5 terminal-failure frozen diagnosis

This bundle freezes the exact failed `scikit-learn__scikit-learn-13135`
matched-tail trajectory and queries `qwen3-coder:30b` at temperature zero.
It separates three questions that an autonomous trajectory alone cannot:

1. Does FULL text reproduce the historical action from the same frozen prefix?
2. Does the matched 95% materialization change that action?
3. Can restoring FULL history at the failed terminal decision repair submission?

FULL reproduces 7/19 historical commands. Matched tail reproduces 6/19 of the
contemporaneous FULL commands. The first repeat-qualified materialization
effect is decision 3: the current oversized source observation falls from
5,757 to 5,468 tokens. FULL and matched-tail each reproduce their own command
on a second query, but the two commands differ. Both are semantically local
reproduction attempts; the shorter one omits secondary diagnostics.

At decision 19, FULL and matched-tail produce exactly the same historical bad
completion command. It emits a nearby source excerpt after the completion
sentinel rather than the required unified diff. Therefore a missing terminal
record or terminal payload is not the cause, and terminal add-back cannot fix
the run. The remaining causal question is whether the repeatable decision-3
payload intervention changes the subsequent autonomous path. That requires a
snapshot-restored rollout from decision 3 and is not answered by frozen
next-action replay.

Artifacts:

- `full_all_a.json`: FULL replay at all 19 decisions against history;
- `full_d3_b.json`: repeat qualification of FULL decision 3;
- `tail_d3_a.json`, `tail_d3_b.json`: repeat qualification of the exact
  matched-tail decision-3 budget;
- `full_d19_a.json`: isolated FULL terminal replay;
- `tail_all_a.json`: matched-tail replay at all autonomous materialized token
  ceilings, compared with contemporaneous FULL;
- `matched_budget_*.json`: frozen materialized-token ceilings;
- `summary.json`: compact reduction and conclusion.
