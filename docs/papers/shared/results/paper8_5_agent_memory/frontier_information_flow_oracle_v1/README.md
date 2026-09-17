# Recent-frontier information-flow DAG oracle screen

This bundle evaluates a boundary-free DAG rule: keep every user instruction,
define the live frontier as the last `M` genuine user prompts, and simplify an
older causal group only when no directed information-flow path reaches that
frontier. The policy sees no evaluator task ID or source-episode boundary.

The graph contains instruction-control, action–observation, next-decision,
same-resource, and declared-dependency edges. Resource identity is scoped by a
generic workspace-lineage ID when available, otherwise by the weaker
environment fingerprint, and finally by an explicit unknown scope. Unknown
tool effects downgrade a no-path result from certified to heuristic.

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

The first paired-FULL-success pilot is Task 5 (`django__django-16145`). FULL
and recent-frontier DAG `M=2` both resolve officially in 12 calls. The DAG run
materializes 137,055 cumulative input tokens versus 264,204 for FULL: 48.13%
paired saving with zero call increase. The runs diverge at the first assistant
action and submit different resolving patches, so this is endpoint-quality
preservation, not exact trajectory parity. It is one task identity and cannot
select a default policy without repetition and additional paired successes.

The Task 6 pilot is an efficiency rejection. Both arms fail official grading,
while DAG `M=2` takes 15 calls versus five and materializes 77.91% more paired
input despite pruning 41.19% relative to its own trajectory counterfactual.
This demonstrates why per-request pruning must not be reported as end-to-end
saving when the policy changes the trajectory.

Complete trajectories, manifests, and official outcomes are in
`autonomous_task5_m2_v1/` and `autonomous_task6_m2_v1/`.

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
