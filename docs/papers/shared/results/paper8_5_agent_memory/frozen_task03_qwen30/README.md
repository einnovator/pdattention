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

`negative_heuristic_structural_exact.json` repeats that opportunity screen
with the pinned Qwen3-Coder tokenizer.  Its counts are cumulative over 23
growing decision prefixes, not unique-history storage savings: H1 `Kf=0`
removes 2,660 of 199,109 tokens (1.34%), H2a 833 (0.42%), H3 `Kr=1` 603
(0.30%), and guarded H2b again abstains.  Aggressive H4 `Kx=2` removes 11,370
(5.71%); `Kx>=3` abstains because dependency pins protect the remaining
resources.

## Seed-0 repeatability and negative-selection cohort

Temperature zero alone was not trajectory-exact on this endpoint.  The
unseeded matched-tail diagnostic is therefore quarantined under `quarantine/`.
`full_seed0_a.json` is the new comparison reference and `full_seed0_b.json`
is its repeatability gate: all 23 generated contents match byte-for-byte.  The
reference's decision 22 is format-invalid in both repeats, so command-rate
denominators deliberately include only the 22 decisions where FULL emitted a
command.

`comparison_seed0.json` and `comparison_seed0.md` contain only arms generated
against that seeded FULL reference.  The first equal-token control shows that
the 603-token certified DAG exclusion and the token-tail control have the same
95.5% exact/conservative action rate and first action divergence at decision
21.  Their cumulative materialized totals differ by only six tokens.  This is
frozen next-action evidence, not autonomous task success.

Immediate H1 (`Kf=0`) is not safe on this trace.  Removing one 190-token
discovery turn after its first downstream read changes content at decision 10,
the exact command at decision 12, and the conservative shell action at decision
13.  Across the trace it saves 2,660 cumulative tokens (1.34%) but preserves
only 68.2% conservative action equivalence, with six immediate resource
reacquisitions and one policy-excess reacquisition.  Delayed H1 and the other
isolated heuristics remain separate treatments rather than being hidden inside
a combined policy.  Delaying H1 to `Kf=4` saves 1,900 cumulative tokens
(0.95%) and moves first conservative action divergence from decision 13 to 16,
but action equivalence reaches only 72.7%; five immediate reacquisitions and one
policy-excess reacquisition remain.  Delay alone therefore does not establish
safe supersession.

`h1_strict_structural_exact.json` evaluates the stricter unresolved-branch
refinement.  It requires every concrete discovered resource to have a complete
successful downstream read/diff before the transition and delay gates apply.
The rule correctly abstains for `Kf=0,1,2,4` on all 23 prefixes because several
discovered branches remain unread.  This supplies no saving on the current
trace and therefore does not require another model replay.

H2a likewise does not qualify as safe exclusion on this trace.  A complete
same-version read/diff permits retirement of one 119-token successful-write
turn, saving 833 cumulative tokens (0.42%), but conservative action equivalence
falls to 77.3% beginning at decision 17.  There are two immediate
reacquisitions, one policy-excess reacquisition, and an action-validity change
at decision 20.  Current file bytes do not preserve all mutation-intent and
progress information carried by the write turn.

Guarded H2b abstains because this trajectory lacks an explicit verification-to-
resource dependency receipt.  Its 23 seeded outputs match FULL exactly and it
saves no tokens.  That validates fail-closed behavior; it does not yet validate
the quality of H2b on a trace where the rule can fire.

H3 with `Kr=1` retires exactly the same old read as the trace-exact DAG
certificate on this trajectory.  Its selected histories and outputs therefore
coincide with DAG-EXCLUDE: 603 cumulative tokens saved, first conservative
action divergence at decision 21, and 95.5% action equivalence.  This trace
cannot distinguish H3's weaker version/span criterion from byte-identical
duplicate elimination.

Aggressive H4 at `Kx=2` saves 11,370 cumulative tokens (5.71%) but first
changes the action at decision 9 and preserves only 59.1% conservative action
equivalence.  It records no immediate reacquisition; that does not imply safe
exclusion because each frozen decision is independent and changed actions need
not explicitly reread the hidden resource.

The first pairwise combinations do not improve the tradeoff.  H1+H3 saves
3,263 cumulative tokens (1.64%) with 59.1% conservative action equivalence;
H2a+H3 saves 1,436 (0.72%) with 77.3%.  Since H2b abstains on every prefix, the
three-rule H1+H3+H2b treatment is materialized-input-equivalent to H1+H3 and is
not rerun as though it were an independent model cohort.

The all-guarded arm at `Kx=4` saves 4,096 cumulative tokens (2.06%) and
preserves 63.6% conservative action equivalence.  H2b and H4 abstain at this
setting, making the realized arm H1+H2a+H3.  Its small action-agreement recovery
relative to H1+H3 despite greater exclusion is non-monotone sensitivity, not a
quality claim.

## Oversized-observation materialization

The real trajectory contains a 2,526-token, 292-line source dump plus later
685- and 737-token observations. `structured_evidence_t512_seed0.json` applies
structured evidence only to tool observations above 512 tokens while retaining
every logical record. Across all 23 decisions it reduces 199,109 cumulative
FULL tokens to 188,072 materialized tokens, a 5.54% saving. It preserves 10/22
exact commands and 12/22 conservatively equivalent shell actions; 22/23
actions are format-valid, exactly matching FULL's validity denominator. A
current whole-record repeat (`full_seed0_c_current.json`) again reproduces all
23 contents and all 22 defined commands, excluding endpoint drift as the cause
of the structured arm's divergences.

For a matched early slice, decisions 3--10, the three materializers show a
clear compression--agreement gradient:

| Materializer | Materialized saving | Exact command | Conservative action |
|---|---:|---:|---:|
| Structured evidence | 4,335 / 57,480 (7.54%) | 4/8 | 4/8 |
| Lexical matched span | 16,248 / 57,480 (28.27%) | 2/8 | 2/8 |
| Fixed head/tail | 17,406 / 57,480 (30.28%) | 1/8 | 1/8 |

All three slice arms emit valid actions at every decision. Structured evidence
is substantially more conservative than either positional control, but it is
not behavior preserving. These remain frozen next-action measurements over an
officially successful source trajectory; they do not establish autonomous task
success under materialization. Machine-readable and Markdown reductions are in
`structured_evidence_t512_comparison.*` and
`materialization_t512_d3_10_comparison.*`.

The audit also found that the two long Django traces are poor 512-token
materialization targets: Task 01 has no oversized tool observation, and Task 02
has one 565-token observation for which structured selection retains every
line. The first no-op run exposed a replay bug: reconstructing all selected
lines with `splitlines()`/join changed trailing bytes while leaving token counts
unchanged. That artifact is quarantined, and the materializer now returns the
canonical observation bytes whenever all lines are selected.
