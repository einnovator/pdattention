# Recent-frontier information-flow DAG oracle screen

This bundle evaluates a boundary-free DAG rule: keep every user instruction,
define the live frontier as the last `M` genuine user prompts, and simplify an
older causal group only when no directed information-flow path reaches that
frontier. The policy sees no evaluator task ID or source-episode boundary.

The graph contains instruction-control, action–observation, next-decision,
same-resource, and declared-dependency edges. Cross-instruction resource
identity requires a generic workspace-lineage ID or an explicit dependency.
An environment/image fingerprint may scope records within one instruction
epoch, but cannot link two epochs because independent containers can share an
image, cwd, and path. Missing lineage makes retirement heuristic rather than
certified. Unknown tool effects also downgrade confidence.

## Structural oracle results

On four constructed six-instruction chains, heuristic no-path retirement has
100% token-weighted precision and recall against the declared dependency
oracle. Plain “drop everything before the last two prompts” fails on the
dependent case: precision is 75% because it incorrectly retires the old
`src/shared.py` lineage. The DAG retains that lineage and reaches 100/100. A
same-path/different-workspace case reaches 100/100 even under the certified
gate, demonstrating that `/testbed` or a common filename must not alias
independent workspaces. In the unknown-effect case, the certified gate fails
closed at 75% recall while heuristic mode reaches 100%; this is intended.

## Frozen real six-issue chain

The real screen uses the completed persistent-FULL six-issue chain
(`qwen3-coder:30b`) and its pinned Qwen tokenizer. The historical traces do not
contain the new workspace-lineage field, so these rows are heuristic, not
certified and not autonomous quality evidence.

| Frontier | DAG vs independent-workspace oracle | Drop whole groups | Omit result payloads | Omit action parameters + results |
|---|---:|---:|---:|---:|
| last 2 user prompts | 100% precision / 100% recall | 39.30% | 18.49% | 33.78% |
| last 3 user prompts | 100% precision / 100% recall | 35.47% | 17.09% | 30.80% |

The denominator is one final 30,190-token frozen history snapshot,
not cumulative autonomous input. The result establishes logical opportunity
and oracle alignment only. It does **not** establish unchanged task accuracy,
calls-to-solution, or end-to-end token saving.

## Autonomous paired pilots

The first paired-FULL-success pilot is Task 5 (`django__django-16145`). Two
recent-frontier DAG `M=2` executions both resolve officially in 12 calls and
save 48.13% and 47.55% paired input (47.84% mean). These are repeated
executions of one identity, not two independent accuracy observations.

Task 3 then exposes a precise missing dependency. Two M2 executions make the
correct source edit but submit prose instead of a unified diff and fail
officially. M2 retains the immediately preceding malformed submission example
from failed Task 2 while retiring the older valid example from Task 1. The P1
repair adds a generic protocol-control edge to the latest harness-certified
valid completion. It is a reachability barrier: its causal bundle remains
visible without pulling the entire completed task back into context. The
repaired Task 3 resolves officially in seven calls. It is individually
inefficient (-4.64% paired saving), but repairs the protocol failure.

The unchanged M2+P1 policy then resolves Task 4 in six calls with 59.28%
paired saving and Task 5 in 12 calls with 47.55% saving. Across the three
distinct paired-FULL-success identities, it preserves 3/3 resolutions, uses
25 calls versus 25, and materializes 280,858 versus 474,697 cumulative input
tokens: 40.83% paired saving. This passes the initial three-identity gate; it
is not an estimate that population accuracy exceeds 90%.

The first repeat exposes why promotion requires contemporaneous paired
controls. Task 3 M2+P1 again resolves, but now takes 13 calls and sends 179,380
tokens. Compared with the historical six-call FULL row this appears to be a
102.67% cost increase. The simultaneously rerun FULL control also resolves in
13 calls and sends 199,232 tokens. The valid repeat comparison is therefore
equal calls and 9.96% saving. P1's quality repair repeats, but the 30--50%
saving gate does not. The first two candidate request inputs match the earlier
repaired run byte-for-byte and generation diverges on request 2, confirming
that temperature zero and fixed seed do not eliminate backend trajectory
variance. The frozen policy remains unpromoted.

Task 4 provides the contrasting stable repeat. Contemporaneous FULL resolves
in seven calls and sends 121,856 tokens, closely matching its historical
7-call/121,983-token control. M2+P1 again resolves in six calls and sends
exactly 49,667 tokens, reproducing the first candidate execution. The valid
paired saving is 59.24% with one fewer call. Thus the current evidence contains
one repeat-qualified high-saving identity and one repeat-qualified low-saving
identity; it does not support a universal 40--50% default.

Task 5 cannot enter the conditional preservation gate because its
contemporaneous FULL repeat fails official grading (10 calls, 211,898 tokens).
The frozen M2+P1 arm resolves in 12 calls with 138,570 tokens, exactly
reproducing its preceding successful candidate run and using 34.61% less input
than the failed FULL trajectory. This is unconditional reliability evidence,
not a preserved-FULL-success point. It motivates randomized repeated pairing
to test whether exclusion reduces harmful cross-task interference.

The sharper boundary-free M1+P1 diagnostic resolves Task 5 officially in two
of two repeats with identical executed-action trajectories. Both use 11 model
calls (10 executed actions) and materialize 109,443 of their own 237,648
counterfactual FULL tokens, a 53.95% within-trajectory saving. Against the
earlier exact-prefix successful FULL row (12 calls, 264,204 tokens), the
descriptive saving is 58.58% with one fewer call. This is a repeated mechanism
point on one task identity, not a population-accuracy claim or a
contemporaneous randomized pair. The repeat ledger is
`autonomous_m1_p1_task5_repeat.json`.

`task5_boundary_free_cross_epoch_audit_m1_p1.json` and
`task5_boundary_free_cross_epoch_audit_m2_p1.json` reconstruct all 12 request
frontiers without evaluator task IDs or episode markers. Both report zero
unexplained cross-epoch edges. M1 retires all 44 causal groups from the four
previous issues and retains one atomic valid-protocol exemplar; M2 also keeps
the immediately preceding issue whole. Therefore the observed Task 5
instability is not caused by an accidental graph path between issues.
The exact prefix and active-episode export consumed by the audit are retained
as `task5_persistent_prefix_before_episode_05.json` and
`autonomous_task5_m2_p1_v1_r03/persistent_episode_export.json`.

M1+P1 transfers successfully to Scikit-learn Task 3 (8 calls; 49,345 of
120,273 counterfactual tokens; 58.97% saving), but fails Pytest Task 4 after
three actions: the agent states the correct edit and submits `git diff` without
mutating the workspace. This narrows the missing state to workflow control,
not a cross-issue resource edge.

The W1 causal add-back retains only the latest successful changed-resource
group, the latest successful observation of that post-mutation resource, and
the harness-certified completion. It restores Task 4 in 2/2 exact-metric
repeats (6 calls; 42,864 of 103,758 tokens; 58.69% saving) and preserves Task 3
(8 calls; 57.15% saving). It then loses Task 5: the model identifies that its
new `display_addr` is absent from the format dictionary but submits before
fixing it. The three-task W1 gate is therefore 2/3 with 56.52% raw
own-trajectory saving and zero failure-aware credit. Raw cross-task workflow
text is not promoted; the next test must use a task-neutral control receipt.
The aggregate and causal diagnosis are in
`autonomous_m1_workflow_spine_three_task_gate.json`.

Across the complete contemporaneous repeat cohort, FULL resolves 2/3 and
M2+P1 resolves 3/3. The arms use 30 and 31 calls and send 532,986 and 367,617
tokens, a descriptive 31.03% input reduction. Conditional on the two FULL
successes, M2+P1 preserves 2/2, uses 19 versus 20 calls, and saves 28.67%
(229,047 versus 321,088 tokens). The conditional estimate is the promotion
metric and narrowly misses the predeclared 30% floor. The unconditional result
is retained separately as reliability evidence.

The Task 6 pilot is an efficiency rejection. Both arms fail official grading,
while DAG `M=2` takes 15 calls versus five and materializes 77.91% more paired
input despite pruning 41.19% relative to its own trajectory counterfactual.
This demonstrates why per-request pruning must not be reported as end-to-end
saving when the policy changes the trajectory.

`autonomous_m2_p1_three_task_gate.json` is the frozen initial aggregate, and
`autonomous_m2_p1_three_task_repeat_cohort.json` is the contemporaneous repeat
aggregate. The
`autonomous_m2_p1_task{3,4,5}_repeat_pair.json` record the contemporaneous
repeats.
Complete
trajectories, manifests, and official outcomes are in the adjacent
`autonomous_task*_m2*` directories. One concurrent Task-5 execution diverged
after several byte-identical selected prompts and caused the grader to exceed
the campaign ceiling; it is retained under `quarantine_task5_m2_p1_v1/` and
excluded from the aggregate rather than silently retried away.

## Implementation correction discovered by this screen

The first audit reached only 16–18% oracle recall because the mini-swe-agent
adapter treated repeated protocol examples (`patch.txt`, `pyproject.toml`, and
similar strings) as task resources. The corrected adapter extracts task paths
from the actual `<pr_description>` envelope. A generic opaque
`workspace_lineage_id`, derived from the live container rather than a task ID,
is now attached to observations for future certified runs.

`evidence.json` contains candidates, confidence levels, false-positive and
false-negative group IDs, simplification costs, source hashes, and all four
constructed cases.
