# Paper 2.5 Pass-1 claim ledger

This ledger maps the manuscript's P0 claims to frozen evidence. The Pass-1
analysis reruns no model or router.

| Claim | Cohort / statistical unit | Evidence | Status and permitted wording |
|---|---|---|---|
| Final-layer native edge R@4 is .428 at `W=16` and .187 under global attention. | Five model seeds per window in the controlled 25-model family. | `controlled_local_sa_v6/receptive_field_topology_summary.csv` | Supported as a controlled receptive-field result. |
| The iterative schedule changes aggregate path recovery by +.0275 and answer accuracy by -.0400. | 400 observed model-example conditions formed by 16 task identities, five windows, and five seeds. Identities, not rows, are the independent resampling clusters. | `controlled_local_sa_v6/traversal_to_use_rows.csv`; `final_reviewer_patch/traversal_effects_p0_summary.csv`; `final_reviewer_patch/traversal_effects_p0_audit.json` | Supported as point estimates for this mechanistic cohort. Identity-clustered 95% intervals are [-.020, .075] and [-.095, .008]. Do not call 400 independent tasks. |
| Path recovery improves in 59 rows, is unchanged in 293, and worsens in 48. | Same 16-identity, 400-row mechanistic cohort. | `final_reviewer_patch/traversal_effects_p0_summary.csv` | Supported as the complete observed outcome distribution. |
| The improved-path subgroup gains 2.164 margin and .102 accuracy; the worse-path subgroup loses 1.585 margin and .292 accuracy. | Post-treatment-selected row groups; membership is held fixed during identity-cluster resampling. | `final_reviewer_patch/traversal_effects_p0_summary.csv` | Conditional descriptive associations only. They do not establish mediation or an aggregate benefit. |
| One-shot and iterative conditions use equal total layer-token K/V state. | Controlled schedules over the same frozen checkpoints and examples. | `controlled_local_sa_v6/local_pra_one_shot_iterative_rows.csv`; `experiments/paper2_5_iterative_pra/run_controlled_pra.py` | Supported as payload matching. The conditions differ in selection state, injection timing, exposure depth, and consumer count; evolving-query updating is not isolated. |
| Executable selected attention divides .147 evidence / .521 distractor / .332 native; oracle divides .270 / .301 / .429. | Mechanistic cohort averaged equally over windows after per-seed aggregation. | `controlled_local_sa_v6/memory_activity_diagnostics.csv` | Supported. Use "increased evidence attention," not "evidence dominated." |
| Oracle memory raises accuracy from .140 to .398. | Frozen-consumer matched intervention on the mechanistic cohort. | `controlled_local_sa_v6/oracle_consumption_ceiling.csv` | Shows room for better selection and bounded consumption capacity; does not establish reliable reasoning. |
| 21.6% of oracle traces with a positive immediate margin effect lose that advantage by the final layer. | Conditional intermediate-readout diagnostic. | `controlled_local_sa_v6/later_layer_erasure.csv` | Describe as reduced intermediate answer decodability. Do not claim that usable information was causally erased. |
| The one-feature retry stump has .638 balanced accuracy on 248 one-shot gold-path misses. | Oracle-defined eligibility subset; leave-one-example-identity-out model selection and grouped bootstrap. | `final_reviewer_patch/iteration_benefit_predictability.json` | Conditional diagnostic only. It is not executable because gold-path miss status is unavailable at inference and all-example retry cost/regressions were not evaluated. |

## Unresolved evidence gaps

- A static-query selector injected on layers 0, 2, 4, and 5 with the same
  no-repeat rule and 20-state budget is needed to isolate residual-dependent
  query updating from the injection schedule.
- A deployable retry evaluation must define eligibility from observable state
  for all examples and report unnecessary retries, regressions, and retrieval
  cost, with selection/tuning confined to training folds.
- More than 16 independent task identities are needed for broad population
  claims about aggregate or selected-subgroup effects.
