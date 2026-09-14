# Paper 2.7: Latent Query Decomposition from Frozen Transformer State Graphs

This paper evaluates whether sparse graph structure over frozen decoder query
states recovers natural query facets and improves matched-budget retrieval.

## Build

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper_2_7.tex
bibtex paper_2_7
pdflatex -interaction=nonstopmode -halt-on-error paper_2_7.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_2_7.tex
```

## Reproduce experiments

From the repository root with `PYTHONPATH=src;.`:

```powershell
python -m pytest tests/test_query_graph.py tests/test_query_graph_cluster.py tests/test_query_graph_facets.py
python -m experiments.paper2_7_query_graph.run_graph_primitives
python -m experiments.paper2_7_query_graph.run_algorithm_cross
python -m experiments.paper2_7_query_graph.run_controlled_facets
python -m experiments.paper2_7_query_graph.run_natural_retrieval --local-files-only
python -m experiments.paper2_7_query_graph.run_encoding_mode_pilot
python experiments/paper2_7_query_graph/build_natural_facet_benchmark.py
python experiments/paper2_7_query_graph/run_llm_decomposition.py --help
python experiments/paper2_7_query_graph/run_natural_facet_validation.py --help
python experiments/paper2_7_query_graph/run_fresh_retrieval.py --help
python experiments/paper2_7_query_graph/summarize_next_iteration.py
```

The natural runner expects inherited frozen feature artifacts documented in
[`experiments/paper2_7_query_graph/README.md`](../../../experiments/paper2_7_query_graph/README.md).
Results are written under
[`../shared/results/paper2_7_query_graph`](../shared/results/paper2_7_query_graph).

The historical study evaluates four requested 32-token chunks. The standalone
iteration adds a fresh four-document endpoint. Its native-K/V gate fails, so it
does not perform physical materialization, memory attention, or answer
generation.
