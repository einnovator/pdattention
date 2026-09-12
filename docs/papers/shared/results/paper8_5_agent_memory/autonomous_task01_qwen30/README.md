# Autonomous Task 01: Qwen3-Coder 30B

This directory contains the first autonomous Paper 8.5 pilot on
`django__django-15277`. All executions use mini-swe-agent 2.4.6,
`qwen3-coder:30b`, the pinned Qwen3-Coder tokenizer, temperature 0, top-p 1,
seed 0, whole-record text materialization, a fresh SWE-bench workspace, and the
official SWE-bench Docker grader.

## Primary and auxiliary outcomes

| Policy | Primary official submission | Auxiliary final workspace | Calls | Full / selected tokens | Saving | Reacq. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL-A | 1/1 | unavailable | 26 | 131,530 / 131,530 | 0.00% | 0 |
| FULL-B | 0/1 | 1/1 | 29 | 154,173 / 154,173 | 0.00% | 0 |
| H3 read supersession, `Kr=1` | 0/1 | 0/1 | 19 | 91,520 / 85,640 | 6.43% | 6 |

The primary endpoint is the official grade of the patch submitted through the
normal mini-swe-agent protocol. The auxiliary outcome is a post-hoc official
grade of a provenance-checked patch extracted from the terminal tracked
workspace. It measures solution state, is labelled auxiliary, and never
replaces or relabels the primary submission outcome.

Token counts are cumulative message-content tokens across growing requests and
exclude chat-template tokens. The different full-history totals are expected
because these are autonomous rollouts: once their actions diverge, their
histories and run lengths differ. H3's lower call count is termination with an
invalid artifact, not a reduction in calls to solution.

FULL-A produced the intended one-line guarded validator change and passed the
primary official grader. Its auxiliary extraction is unavailable because the
terminal checkpoint contains the untracked `patch.txt`; the extractor fails
closed rather than silently discarding untracked state. FULL-B produced the
same intended edit but, after its action path diverged from FULL-A at call 9,
finalized by printing a source excerpt rather than a unified diff. Its primary
submission therefore fails. The auxiliary 749-byte tracked-workspace patch
(`01522b061b86c4b974793f938c2a9764b5feff09dff7d2f9c788d29eb390bf2d`)
resolves officially. FULL-B is therefore a submission-protocol failure, not a
solution-state failure.

The primary FULL endpoint is still 1/2 on this task, and the two trajectories
remain unstable in calls and actions. The auxiliary result narrows the failure
mechanism but does not make the autonomous trajectories repeatable or change
the primary denominator.
A separate fixed first-request probe is byte-identical in 3/3 assistant
contents and commands; that narrow check does not qualify the growing
autonomous trajectory, where generated reasoning and environment observations
feed the next request.

The corrected H3 run has an exact no-op control: its canonical input history
and requests match FULL-B through call 5. At call 6 H3 first excludes an older
same-resource read bundle; the assistant reasoning changes immediately while
the shell action remains the same. The command trajectory diverges at call 7.
In the subsequent trajectory, H3 repeatedly
reacquires the same source, then applies an unscoped global `sed` replacement.
That replacement adds the intended guard to `CharField` but also rewrites the
already guarded `BinaryField`, producing a malformed duplicate `if`. The final
submission command emits broad `grep` output rather than `git diff`, so the
captured `model_patch` is not a valid patch. Unlike FULL-B, the auxiliary H3
workspace also fails: its 1,090-byte patch
(`b1171e2c84ff57d3f9f3f31985f559489016100c112c3af0624153c8aba2694d`)
contains the malformed duplicate guard and scores 0/1.

This is a single-task pilot, not an estimate of H3's population success rate.
It is a reproducible next-action counterexample to treating same-version/span
read supersession as behaviorally inert: the model-visible H3 trajectory was
identical in two executions, including its failure mechanism, and the qualified
run's first reasoning divergence coincides with the first real exclusion. It
does not by itself estimate a population penalty. The auxiliary comparison does
separate mechanisms: FULL-B reaches a resolving state but violates submission
protocol, whereas H3 fails under both outcome definitions. A later read can
supersede observed bytes without establishing that the earlier reasoning,
localization, or mutation intent is irrelevant. H3 remains a heuristic arm;
certified exclusions continue to fail closed when their evidence requirements
are not met.

## Validity and provenance

Earlier pilot attempts with an invalid instrumented environment or a changed
request envelope during policy no-op decisions were quarantined and are not
included here. The proxy now forwards the exact request whenever selection is
a logical no-op; the included H3 run records five exact pass-through decisions
before its first exclusion. Each included run records the exact model, tokenizer, decoding settings,
harness and grader versions, dataset revision, benchmark-card hashes, commands,
and repository revision in `run_manifest.json`.

## Files

`comparison.json` is the compact cross-run reduction, and
`static_request_repeatability.json` records the 3/3 fixed-request diagnostic.
Each `auxiliary_workspace_state*.json` file records availability, hashes,
grading provenance, and the explicit `auxiliary_only` evidence role.

Each execution contains:

- `autonomous_metrics.json`: calls, token accounting, exclusions, and
  reacquisition counters;
- `official_result.json`: official SWE-bench result;
- `official_raw_report.json`: copied grader report (the path inside
  `official_result.json` records the execution host);
- `request_selection.jsonl`: per-request selected records and rule receipts;
- `trajectory.json`: complete mini-swe-agent interaction history;
- `run_manifest.json`: frozen execution provenance.

The H3 and FULL-B directories additionally contain `preds.json`, exposing the
non-diff submissions received by the official grader. FULL-A also includes
`negative_structural.json`, a structural diagnostic that does not alter its
requests.
