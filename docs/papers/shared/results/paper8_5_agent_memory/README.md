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

## First instrumented frozen cohort

The later [`frozen_task03_qwen30`](frozen_task03_qwen30/README.md) cohort uses a
new 23-decision successful Task 03 trajectory with cwd, completeness,
environment, workspace-checkpoint, resource-version, and traced-tool-semantics
evidence.  It uses the exact model tokenizer rather than whitespace counts and
compares generated next actions with a contemporaneous FULL replay.  Its named
90% treatments use a minimum-retention floor with whole-turn round-up; this is
a different budget contract from the ceiling-based structural pilot above.
Its directory also contains a separately labelled whitespace-only H1--H4
opportunity screen; the heuristic rows must not be interpreted as frozen or
autonomous quality evidence.

## First autonomous pilot

The [`autonomous_task01_qwen30`](autonomous_task01_qwen30/README.md) cohort is
the first fresh-workspace mini-swe-agent execution with official SWE-bench
grading. The primary full-history submission resolves one of two repeated
executions. H3 with
`Kr=1` saves 6.43% of cumulative message-content tokens, changes reasoning at
the first real exclusion and the command at the following call, reacquires
excluded state six times, and does not produce a valid final patch. Because a
full-history repeat also fails, this is evidence that H3 is not behaviorally
inert, not yet an estimate of its population task-success penalty. A separate
auxiliary grade of terminal tracked-workspace state resolves FULL-B but not H3:
the former is a submission-protocol failure, whereas H3's workspace contains
the malformed duplicate guard. The auxiliary grade is explicitly secondary and
does not replace the primary official submission outcome. FULL-A auxiliary
extraction fails closed because its checkpoint contains untracked `patch.txt`.

## Replicated FULL-control instability

The [`autonomous_task02_qwen30`](autonomous_task02_qwen30/README.md) cohort adds
two FULL-history controls on a second task identity,
`django__django-15368`. FULL-A resolves in 30 calls; FULL-B stops after 13 calls
and submits a source excerpt rather than a unified diff, producing a primary
official patch-application error. All 43 requests are exact pass-through, both controls
start from the same image and exact workspace/environment fingerprints, and
there are no selection exclusions, upstream errors, or reacquisitions.

The requests remain identical through call 5. On the identical call-5 request,
assistant content differs while the command remains equal; the accumulated
requests and commands first differ at call 6. Task 02 therefore independently
repeats Task 01's 1/2 FULL outcome. Temperature zero and a fixed seed describe
the decoding configuration but do not establish deterministic autonomous
trajectories. The auxiliary 731-byte final-workspace patch from FULL-B resolves
officially, showing that its primary failure is submission protocol rather than
solution state. FULL-A auxiliary extraction is unavailable because untracked
`bulk_update_fix.patch` and `test_fix.py` make a tracked-only reconstruction
incomplete. The predeclared gate required both primary Task 02 FULL submissions
to solve with sufficiently stable behavior, so it failed and no Task 02
heuristic arm was run. Certified and strict exclusions remain fail-closed when
their evidence gates are not met.

## Long-horizon and high-retention screen

The expanded structural screen uses the two ordinary-text FULL executions that
both resolve officially and satisfy the predeclared long-run thresholds of at
least 25 model decisions and 100,000 cumulative exact-tokenizer input tokens:

- `django__django-15277`: 26 decisions;
- `django__django-15368`: 30 decisions.

[`long_horizon_structural_screen_qwen30_exact.json`](long_horizon_structural_screen_qwen30_exact.json)
sweeps `Kf={0,2,4,8,12,16}`, `Kw={1,2,4,8,12}`,
`Kr={1,2,3,4,6,8}`, and `Kx={2,3,4,6,8,12}` using the pinned
Qwen3-Coder tokenizer. It also reports aggregates for the complete run and
suffixes beginning at decisions 10, 15, and 20. A bug in the earlier grid
driver was corrected: combination policies now sweep every parameter used by
their constituent rules rather than varying `Kf` while silently fixing `Kr`
and `Kx`.

The 56 growing prefixes contain 284,848 cumulative exact tokens; the 18
decision-20-and-later prefixes contain 127,332. Structural opportunity is
larger late in the run:

| Treatment | Whole-run saving | Decision 20+ saving |
|---|---:|---:|
| H3, `Kr=1` | 6.23% | 8.04% |
| H3, `Kr=2` | 0.73% | 1.63% |
| H1 `Kf=8` + H3 `Kr=2` | 1.24% | 2.78% |
| H1 `Kf=12` + H3 `Kr=2` | 0.83% | 1.86% |

The `Kf=8, Kr=2` combination is the first conservative high-retention
candidate with opportunity on both task identities: 98.76% aggregate
retention over the full runs and 97.22% after decision 20. H1 supplies the
Task 01 exclusions, while H3 supplies Task 02 exclusions. `Kr>=4` and `Kx>=6`
abstain on these traces, so simply increasing every `K*` eventually eliminates
the treatment.

[`long_horizon_high_retention_qwen30_exact.json`](long_horizon_high_retention_qwen30_exact.json)
screens 100%, 99%, 97.5%, 95%, and 90% token ceilings with tail floors from 2
through 20. A fixed keep-last-20 policy retains 95.77% on Task 01's late
suffix but only 86.38% on Task 02's. This confirms that a common turn count is
not a matched retention control. Model-facing comparisons must instead use the
semantic arm's exact per-decision materialized-token count for the blind token
tail.

These two exact-tokenizer artifacts remain structural opportunity evidence.
They do not establish next-action agreement or autonomous success. The two
`*_whitespace.json` companions are implementation cross-checks only.
