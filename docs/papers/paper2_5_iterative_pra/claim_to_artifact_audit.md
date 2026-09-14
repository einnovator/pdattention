# Paper 2.5 Claim-to-Artifact Audit

This audit maps the reviewer-facing headline claims in `paper.tex` to canonical public artifacts.
Values in the abstract, main tables, and conclusion were checked against these files before the
final build. Private behavioral-judge truth and unblinding files are intentionally excluded.

| Headline claim | Canonical artifact | Primary fields or rows |
|---|---|---|
| Across the 16-identity mechanistic cohort, the iterative schedule changes path recovery by +.0275 and answer accuracy by -.0400; path improves in 59/400 rows, is unchanged in 293, and worsens in 48. | `../shared/results/paper2_5_iterative_pra/final_reviewer_patch/traversal_effects_p0_summary.csv`, `traversal_effects_p0_audit.json`, and `controlled_local_sa_v6/traversal_to_use_rows.csv` | Point estimates and example-identity-clustered intervals over 16 identities crossed with 25 window-seed models. The 400 rows are not independent tasks. |
| The path-improved minority is evidence poor after one shot; among one-shot misses it has shorter chains, longer evidence spans, lower evidence mass, and higher distractor mass. | `../shared/results/paper2_5_iterative_pra/final_reviewer_patch/iteration_benefit_feature_summary.csv` | `one_shot_miss_only` pre-decision rows with means, medians, standardized effects, and bootstrap intervals. |
| A grouped label-free pre-decision stump is not deployment-ready: balanced accuracy .638, recall .593, and precision .368 over 16 held-out task identities. A query-length-only sensitivity reaches .839 balanced accuracy because the synthetic generator couples length to chain depth. | `../shared/results/paper2_5_iterative_pra/final_reviewer_patch/iteration_benefit_predictability.json` | All 16 leave-one-example-identity-out folds, candidate features, selected thresholds, pooled confusion counts, identity-bootstrap intervals, and the separately labeled generator-coupled sensitivity. |
| Hotpot 4-token contextual facets raise selected-root inclusion from .343 to .457 at 20%, while query-parent comparisons rise from 5.9 to 67.6. | `../shared/results/paper2_5_iterative_pra/query_entry_facets/query_entry_summary.csv` and `final_reviewer_patch/dataset_routing_geometry_summary.csv` | Held-out `A_global_semantic` and `B_multi_span_semantic` rows. |
| QASPER/Hotpot/2Wiki/MuSiQue differ in root entry, evidence dispersion, successor topology, and conceptual payload. | `../shared/results/paper2_5_iterative_pra/final_reviewer_patch/dataset_routing_geometry_summary.csv` | Exact metrics copied from canonical frozen summaries plus explicitly labeled qualitative geometry. |
| Final-layer controlled edge R@4 declines from .428 at W=16 to .187 globally while shortcut rate rises from .429 to .584. | `../shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/receptive_field_topology_summary.csv` | Five-window final-layer means, SDs, intervals, and empirically best layers. |
| The iterative schedule improves traversal more consistently than answer accuracy; every five-seed model-level accuracy interval crosses zero. | `../shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/iterative_matched_budget_by_window.csv` and `paired_pra_effects.csv` | Exact 20-state one-shot/iterative rows and depth-macro paired seed effects. Equal payload does not match selection state, timing, exposure depth, or consumer count. |
| In the mechanistic cohort, path-improved rows gain +2.164 correct-label margin and +.102 accuracy, while path-worsened rows lose 1.585 and .292. | `../shared/results/paper2_5_iterative_pra/final_reviewer_patch/traversal_effects_p0_summary.csv` | Post-treatment-selected descriptive groups. Membership is fixed during identity-cluster resampling; these are conditional associations, not mediation estimates. |
| Matched oracle iterative memory raises accuracy from .140 to .398 and mean margin from -3.103 to -.703; wrong memory lowers accuracy to .058. | `../shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/oracle_consumption_ceiling.csv` and `causal_memory_paired_effects.csv` | Five-window condition means and model-seed paired deltas relative to no memory. This shows bounded consumption capacity, not reliable reasoning. |
| Ordinary iterative attention divides .147 evidence/.521 distractor/.332 native; oracle divides .270/.301/.429. | `../shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/memory_activity_diagnostics.csv` | Exact evidence/distractor/native shared-softmax decomposition; mass sums equal one. Oracle increases evidence attention but remains distractor- and native-heavy. |
| Oracle usefulness is strongest at early consumers: layer-0 accuracy is .358 versus .138 at layer 5. | `../shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/consumer_layer_profile.csv` | Oracle-only matched payload by window and consumer layer. |
| 21.6% of oracle traces with a positive intermediate margin effect lose that answer-decoding advantage by the final layer. | `../shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/later_layer_erasure.csv` | Intermediate and final paired margin effects by example, seed, window, and policy. This is reduced answer decodability, not proof that usable information was erased. |
| Synthetic two- and three-hop discovery recovers every indirect node at matched final budget. | `../shared/results/paper2_5_iterative_pra/local_associative_closure/gate2_local_results.json` | Synthetic aggregate rows for local bridge controls; exact identity, chain completion, and coverage. |
| At 128-token nodes, held-out 2Wiki transition R@4/R@6/R@8 is .720/.880/1.000 and path survival is .588/.824/1.000. | `../shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_depth_results.json` | 2Wiki transition and path-survival summaries at 128 tokens. |
| MuSiQue/2Wiki complete evidence recovery reaches .981/.979 while selecting .513/.770 of source. | `../shared/results/paper2_5_iterative_pra/final_metrics/cross_dataset_summary.csv` | Dataset headline rows; complete recovery and selected-source fraction. |
| At 16-token nodes, selected source falls to .063/.172 and complete recovery falls to .500/.583. | `../shared/results/paper2_5_iterative_pra/natural_graph_depth/cross_dataset_granularity.csv` | MuSiQue and 2Wiki 16-token rows. |
| Native topology varies non-monotonically by layer and node size; 2Wiki R@6 is .880 at layers 0/27 and .920 at layers 12/20. | `../shared/results/paper2_5_iterative_pra/layerwise_graph/layer_granularity_summary.csv` | 2Wiki 128-token layer rows and layer-by-granularity maxima. |
| Contextual dependence does not determine graph quality. | `../shared/results/paper2_5_iterative_pra/final_metrics/layer_context_correlations.csv` | Correlations between contextualization measures and edge/path metrics. |
| All-offset oracle root Top-4 is .878/.950 on MuSiQue/2Wiki, versus .622/.692 for held-out executable selectors. | `../shared/results/paper2_5_iterative_pra/natural_graph_depth/routing_ceiling_table.csv` | Oracle-facet and bounded/executable held-out rows. |
| Gate 3 contains 1,008 frozen generation runs, enforces a 256-token native-operation bound, and records zero violations. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_generation_results.json` | Run count, `max_native_operation_tokens`, and violation totals. |
| Main output timing reports synchronized TTFT, TPOT, and total generation latency for each frozen condition. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_summary.csv` | `ttft_seconds`, `tpot_seconds`, and `total_generation_seconds`; displayed values are rounded to two decimals. |
| Held-out 2Wiki one-shot/balanced/high-recall/oracle F1 is .354/.313/.271/.362. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_summary.csv` | 2Wiki held-out policy rows, answer F1. |
| Broad 2Wiki discovery recovers all evidence but final-token attention assigns .044 to evidence and .754 to selected non-evidence. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_analysis.json` | 2Wiki high-recall evidence recovery and attention-mass decomposition. |
| MuSiQue normalized answer accuracy is zero even with direct full context. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_summary.csv` | MuSiQue direct-full-context row. |
| The frozen output policies do not establish a statistically resolved quality improvement over one-shot. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_paired_bootstrap.csv` | Paired mean deltas and identity-bootstrap confidence intervals. |
| A control-qualified judge rates 2Wiki balanced vs one-shot at 90.0 equivalence and +2.9 relative quality, while selected-band vs all-layer is only 20.4 equivalent and favors the sparse band by +37.4. | `../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/behavioral_judge_external_metrics.json` | Complete GPT-5.6 Sol aggregates after collapsing 294 A/B-reversed pairs; these contrasts are not conflated. |
| Claude is not used for efficacy interpretation. | `../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/behavioral_judge_external_metrics.json` | 51.7% pair coverage; failed identical/corrupted calibration; .270 shared-pair direction agreement. |

## Claim Boundaries

- The main causal interpretation is `associative topology -> iterative traversal -> controlled
  activation -> consumption -> preservation`. These stages name different measured events.
- The path-improved subgroup has a positive conditional association with margin, while aggregate
  answer accuracy changes by -.0400 and worse-path rows are harmful. The selected subgroup cannot
  establish an overall benefit or causal mediation.
- `G+` membership structurally requires a one-shot gold-path miss. The fitted stump is evaluated only
  on that oracle-defined subset, whose eligibility is unavailable at inference. It is a conditional
  diagnostic, not a deployable retry policy; all-example retries, regressions, and cost are unmeasured.
- One-shot measurements are pre-decision only with respect to choosing an iterative retry. Iterative
  attention, margin gain, answer gain, and erasure are post-treatment diagnostics.
- Oracle memory is a matched causal consumption control. It demonstrates bounded frozen-consumer
  response capacity but neither reliable reasoning nor an executable routing policy.
- Distractor-dominated attention identifies controlled activation as the dominant measured
  limitation. Physical K/V disclosure and PRA-aware training remain separate paper boundaries.
- Evidence reachability is not faithful traversal of the annotated reasoning graph.
- Oracle-root results are ceilings, not end-to-end routing results.
- Selected-source fraction is conceptual activation, not physical native-K/V materialization.
- Gate-3 active K/V plus synchronized TTFT, TPOT, and total latency are frozen single-example
  efficacy measurements, not throughput, concurrency, or production-serving benchmarks.
- Five router seeds measure projection sensitivity; they do not multiply independent question
  identities.
- MuSiQue output comparisons are backbone-limited because the direct-full-context control also has
  zero normalized accuracy.
- Behavioral scores are evaluator measurements, not ground truth. Only GPT-5.6 Sol passes the
  preregistered calibration anchors; Claude is retained as a failed-instrument diagnostic.
- Parameter directions are synthesis claims, not monotonic laws. The frozen source map is
  `final_reviewer_patch/pra_parameter_directionality.md`.
