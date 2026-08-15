# Paper 2.5 Claim-to-Artifact Audit

This audit maps the reviewer-facing headline claims in `paper.tex` to canonical public artifacts.
Values in the abstract, main tables, and conclusion were checked against these files before the
final build. Private behavioral-judge truth and unblinding files are intentionally excluded.

| Headline claim | Canonical artifact | Primary fields or rows |
|---|---|---|
| Synthetic two- and three-hop discovery recovers every indirect node at matched final budget. | `../shared/results/paper2_5_iterative_pra/local_associative_closure/gate2_local_results.json` | Synthetic aggregate rows for local bridge controls; exact identity, chain completion, and coverage. |
| At 128-token nodes, held-out 2Wiki transition R@4/R@6/R@8 is .720/.880/1.000 and path survival is .588/.824/1.000. | `../shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_depth_results.json` | 2Wiki transition and path-survival summaries at 128 tokens. |
| MuSiQue/2Wiki complete evidence recovery reaches .981/.979 while selecting .513/.770 of source. | `../shared/results/paper2_5_iterative_pra/final_metrics/cross_dataset_summary.csv` | Dataset headline rows; complete recovery and selected-source fraction. |
| At 16-token nodes, selected source falls to .063/.172 and complete recovery falls to .500/.583. | `../shared/results/paper2_5_iterative_pra/natural_graph_depth/cross_dataset_granularity.csv` | MuSiQue and 2Wiki 16-token rows. |
| Native topology varies non-monotonically by layer and node size; 2Wiki R@6 is .880 at layers 0/27 and .920 at layers 12/20. | `../shared/results/paper2_5_iterative_pra/layerwise_graph/layer_granularity_summary.csv` | 2Wiki 128-token layer rows and layer-by-granularity maxima. |
| Contextual dependence does not determine graph quality. | `../shared/results/paper2_5_iterative_pra/final_metrics/layer_context_correlations.csv` | Correlations between contextualization measures and edge/path metrics. |
| All-offset oracle root Top-4 is .878/.950 on MuSiQue/2Wiki, versus .622/.692 for held-out executable selectors. | `../shared/results/paper2_5_iterative_pra/natural_graph_depth/routing_ceiling_table.csv` | Oracle-facet and bounded/executable held-out rows. |
| Gate 3 contains 1,008 frozen generation runs, enforces a 256-token native-operation bound, and records zero violations. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_generation_results.json` | Run count, `max_native_operation_tokens`, and violation totals. |
| Held-out 2Wiki one-shot/balanced/high-recall/oracle F1 is .354/.313/.271/.362. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_summary.csv` | 2Wiki held-out policy rows, answer F1. |
| Broad 2Wiki discovery recovers all evidence but final-token attention assigns .044 to evidence and .754 to selected non-evidence. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_analysis.json` | 2Wiki high-recall evidence recovery and attention-mass decomposition. |
| MuSiQue normalized answer accuracy is zero even with direct full context. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_summary.csv` | MuSiQue direct-full-context row. |
| The frozen output policies do not establish a statistically resolved quality improvement over one-shot. | `../shared/results/paper2_5_iterative_pra/output_validation/gate3_output_paired_bootstrap.csv` | Paired mean deltas and identity-bootstrap confidence intervals. |
| A control-qualified judge rates 2Wiki balanced vs one-shot at 90.0 equivalence and +2.9 relative quality, while favoring selected-band over all-layer output by +37.4. | `../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/behavioral_judge_external_metrics.json` | Complete GPT-5.6 Sol aggregates after collapsing 294 A/B-reversed pairs. |
| Claude is not used for efficacy interpretation. | `../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/behavioral_judge_external_metrics.json` | 51.7% pair coverage; failed identical/corrupted calibration; .270 shared-pair direction agreement. |

## Claim Boundaries

- Evidence reachability is not faithful traversal of the annotated reasoning graph.
- Oracle-root results are ceilings, not end-to-end routing results.
- Selected-source fraction is conceptual activation, not physical native-K/V materialization.
- Gate-3 active K/V and latency are frozen efficacy measurements, not TTFT, TPOT, throughput, or
  concurrent-serving benchmarks.
- Five router seeds measure projection sensitivity; they do not multiply independent question
  identities.
- MuSiQue output comparisons are backbone-limited because the direct-full-context control also has
  zero normalized accuracy.
- Behavioral scores are evaluator measurements, not ground truth. Only GPT-5.6 Sol passes the
  preregistered calibration anchors; Claude is retained as a failed-instrument diagnostic.
