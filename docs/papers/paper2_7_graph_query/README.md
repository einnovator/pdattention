# Paper 2.7: Query-Graph Facet Discovery

This paper evaluates sparse graph structure over frozen query representations as
an opt-in facet generator for the unchanged Paper 2.6 PRA memory scorer.

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
```

The natural runner expects inherited frozen feature artifacts documented in
[`experiments/paper2_7_query_graph/README.md`](../../../experiments/paper2_7_query_graph/README.md).
Results are written under
[`../shared/results/paper2_7_query_graph`](../shared/results/paper2_7_query_graph).

The study evaluates retrieval selections at four requested 32-token chunks. It
does not perform physical native-K/V materialization, memory attention, answer
generation, or LLM judging.
