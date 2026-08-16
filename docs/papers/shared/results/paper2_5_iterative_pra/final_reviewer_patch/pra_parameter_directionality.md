# PRA Parameter Directionality

These are dominant directions, not universal monotonic laws. Effects depend on dataset, layer, granularity, and routing quality.

| Parameter increase | Root recall | Path recall | Distractor load | Active K/V | Search cost | Typical benefit |
|---|---:|---:|---:|---:|---:|---|
| `F` query facets | up | indirectly up | up | indirectly up | up | ambiguous or multi-intent queries |
| `R` roots | up | up | up strongly | up | up | uncertain root rank |
| `K` neighbors | unchanged | up | up | up | up strongly | noisy successor ranking |
| `H` hops | unchanged | up for deep paths | up strongly | up | up strongly | distributed evidence |
| `B` final budget | up or unchanged | up | up strongly | up strongly | unchanged or up | incomplete coverage |
| `theta` threshold | precision up, recall down | varies | down | down | down | confidence filtering |
| PRA consumer layers | unchanged | potentially up | interference possible | up strongly | up | repeated assimilation |
| Finer chunks | varies | edge recall can fall | payload down | down per node | node count up | precise disclosure |

Artifact anchors: `query_entry_facets/query_entry_summary.csv`, `natural_graph_depth/natural_graph_depth_results.json`, `natural_graph_depth/cross_dataset_granularity.csv`, `controlled_local_sa_v6/oracle_consumption_ceiling.csv`, and `controlled_local_sa_v6/consumer_layer_profile.csv`.
