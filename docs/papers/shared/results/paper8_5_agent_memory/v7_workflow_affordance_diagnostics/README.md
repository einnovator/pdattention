# Workflow-affordance diagnostics

This bundle decomposes the Task-7 action-cost regression reported in the
`v6_static_workflow_anchor_v2` bundle.  Every model request uses the locked
`qwen3-coder:30b-ctx131k` endpoint with temperature 0, top-p 1, top-k 1 and
seed 0.  Stopped runs are mechanism diagnostics and have no official outcome.

## Results

| Treatment | Evidence | Calls observed | Token evidence | Result |
|---|---|---:|---:|---|
| Guarded prose v3, FULL | autonomous | 11 | 393,714 selected | official solve, but five calls worse than anchored v2 FULL |
| Guarded prose v3, P0/W0 | stopped autonomous | 18 | 98,414 / 698,678 selected/full; 85.91% own saving | repeated malformed `sed` edit loop; no terminal claim |
| v2 plus one old workflow exemplar (`W1`) | stopped autonomous | 8 | 38,191 / 285,791; 86.64% own saving | model incorrectly reasoned about a prior pytest workspace while running Django |
| Compact state-machine v4 | frozen requests | 3 repeats per state | n.a. | stable `ls -la` first action and the same unsafe range-`sed`; rejected |
| Transactional-replace v5 | frozen requests | 3 repeats per state | n.a. | stable filename search at request 1; correct `pra_replace` edit in 3/3 at the frozen mutation state |
| Content-search example v6 | frozen requests | 3 repeats per state | n.a. | invalid multiple-action first response in 3/3; correct `pra_replace` edit in 3/3; rejected |
| v5 plus installed `pra_replace`, P0/W0 | stopped autonomous | 16 | 51,316 / 584,884; 91.23% own saving | transactional edit succeeds once, but localization consumes calls 1--8 and verification continues through call 16 |

The v5 autonomous run was stopped without submission, grading or a capability
claim.  Its execution-9 receipt proves that `pra_replace` returned zero and
changed exactly the intended file version.  The run also exposed a metadata
bug: the generic path regex treated `self.dict` inside replacement text as a
second resource.  Commit `7284f0ee` fixes the adapter to declare only the first
path argument and makes a successful transactional replacement a complete,
runtime-traced write rather than an unknown-effect barrier.  That post-fix
metadata still requires a fresh run before it can support a DAG-selection
claim.

## Interpretation

The action-cost increase has at least three separable causes:

1. **Localization:** retiring old history changes confidence and adds search
   and inspection calls even when every active-task record is retained.
2. **Mutation:** Bash `sed` is fragile.  A typed, uniqueness-guarded replacement
   removes the repeated edit/revert loop, but it does not repair localization.
3. **Termination:** after a successful mutation, the model can continue with
   redundant edge-case and test-runner probes.

An old workflow K/V exemplar is not a task-neutral repair because those vectors
were contextualized under the prior task and workspace.  Likewise, prose that
describes an entire workflow can induce the model to emit several actions at
once.  Tool capability, selection policy and stopping policy must therefore be
evaluated as separate coordinates.  No treatment in this bundle is promoted
to a default profile.

## Provenance

- Guarded-v3 implementation: `440dd4df`.
- Frozen counterfactual runner: `b9784bfe`.
- Transactional tool implementation: `2b72596e` and `812b2b55`.
- Corrected transactional resource receipt: `7284f0ee`.
- Reference anchored-v2 FULL/P0-W0 pair: `v6_static_workflow_anchor_v2`.
