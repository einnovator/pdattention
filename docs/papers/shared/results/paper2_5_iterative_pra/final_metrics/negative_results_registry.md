# Paper 2.5 Negative-Results Registry

This registry freezes measured boundaries; it is not a list of failed software runs.

| Finding | Canonical artifact |
|---|---|
| Iterative parent closure is not reliably above matched one-shot retrieval. | `../iterative_closure_aggregate.csv` |
| Static query reranking hurts under the matched controlled gate. | `../monotonic_adaptive_competition/heldout_effects_vs_one_shot.csv` |
| Dynamic Q/A reconstruction fails its predeclared improvement gate. | `../dynamic_query_discovery/dynamic_query_gate_results.json` |
| Terminal max-facet threshold does not calibrate robustly. | `../semantic_graph_search/goal_threshold_audit.csv` |
| Facet complementarity does not separate plausible false terminals. | `../semantic_graph_search/false_goal_review.csv` |
| Native head splitting is inconsistent across datasets. | `../query_entry_facets/head_selection_audit.csv` |
| Fine chunks reduce native annotated-edge quality despite lower payload. | `../natural_graph_depth/transition_path_by_granularity.csv` |
| Measured contextualization alone weakly predicts graph quality. | `../layerwise_graph/layerwise_correlations.csv` |
| Executable facet competition remains far below the oracle all-offset ceiling. | `../natural_graph_depth/routing_ceiling_table.csv` and `winning_facet_summary.csv` |

The final gate adds no learned production router, adaptive layer fusion, adaptive search
budget, terminal stopping rule, or materialization mechanism. Those are deferred to Papers 3
and 3.5.
