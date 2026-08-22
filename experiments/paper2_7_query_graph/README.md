# Paper 2.7 query-graph experiments

This directory evaluates query-side facet discovery while keeping Paper 2.6
memory discovery and its four-chunk conceptual budget fixed. The graph path is
opt-in and does not alter model defaults, memory traversal, or K/V
materialization.

Run from the repository root with `PYTHONPATH=src;.` on Windows:

```powershell
python experiments/paper2_7_query_graph/run_graph_primitives.py
python experiments/paper2_7_query_graph/run_algorithm_cross.py
python experiments/paper2_7_query_graph/run_controlled_facets.py
python experiments/paper2_7_query_graph/run_natural_retrieval.py --local-files-only
python experiments/paper2_7_query_graph/run_encoding_mode_pilot.py
```

The natural runner reads frozen Paper 2.5 query states and Paper 2.6 memory
features. It reproduces archived Paper 2.6 routes before evaluating graph
facets. Physical K/V materialization and generation are outside this study.
