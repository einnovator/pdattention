# Task 03 frozen ordinary-text cohort

This directory contains the first completed instrumented Paper 8.5 frozen
cohort for `scikit-learn__scikit-learn-13135`, using `qwen3-coder:30b`, the
pinned Qwen3-Coder tokenizer, temperature zero, and the same saved decision
prefixes.  It measures next-action sensitivity, not autonomous SWE-bench task
success and not native K/V performance.

`full.json` is the contemporaneous reference.  `dag_exclude_100.json` uses the
`TRACE_EXACT_OPERATION_RESULT_V1` operational certificate.  The 90% arms treat
the requested percentage as a floor and round upward at whole causal-turn
boundaries.  `matched_causal_tail.json` is deliberately a hard-ceiling causal
control; it demonstrates that whole-turn granularity cannot closely match the
small DAG exclusion on this trace.  A token/within-observation matched control
is therefore required before comparing exclusion quality at equal context.

Headline observations from `comparison.json`:

- FULL is reproducible against itself; its decision 22 response is
  format-invalid in every identical-history control.
- Certified DAG exclusion removes 603 cumulative tokens (0.3%), changes the
  actions at decisions 21--22, and returns to the exact final submission action
  at decision 23.
- Causal recency at the DAG ceiling removes 7,800 tokens because its indivisible
  turn cannot use 7,197 tokens of the matched allowance; its 77.0% minimum
  retention is not a valid equal-token control for the 98.2% DAG decisions.
- Nominal recency-90 retains the entire trace: the oldest large turn is required
  to cross the floor.  Its cumulative whole-turn overshoot is 19,899 tokens.
- Lexical-90 realizes 94.7% mean retention and 5.9% cumulative logical saving.
  Exact command agreement is 45.5%, while conservative action equivalence is
  59.1%; every generated action is format-valid and the final action is exact.
- Progress-spine-plus-lexical realizes 95.5% mean retention and 5.2% saving.  It
  delays first divergence from decision 6 to decision 8, but introduces a
  format-invalid response at decision 9 and has the same 59.1% conservative
  action-equivalence rate.  Its final action is exact.
- DAG-first spine-plus-lexical produces the same frozen responses as the plain
  spine-plus-lexical arm, but retains 95.7% and saves only 4.9%.  Removing the
  certified duplicate changes which whole turn crosses the 90% floor, so the
  fallback adds a larger replacement.  Two-stage exclusion plus round-up is
  therefore not monotone in realized saving.

`comparison.md` is the concise human table; `comparison.json` is the
machine-readable reduction.  `trajectory.json` and each arm preserve the raw
messages, selections, generated responses, token accounting, and digests.

`negative_heuristic_structural_whitespace.json` is a no-model opportunity
screen over H1--H4 and their first combinations. It deliberately uses the
whitespace counter, `head=0`, and two protected tail turns to expose whether
each rule fires; it is not part of the tokenizer-exact frozen comparison.
H1 is the largest guarded opportunity on this trace (1.5% cumulative
whitespace-token saving at `Kf=0`), H2a removes 0.3%, H3 `Kr=1` removes 0.2%,
and guarded H2b abstains because verification-resource dependency metadata is
absent. H4 `Kx=2` reaches 3.5% but is explicitly an aggressive working-set
heuristic. No next-action or task-success result is attached to these rows.
