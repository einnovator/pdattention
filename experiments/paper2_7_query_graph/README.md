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

## Standalone natural-validation iteration

The next iteration adds a source-preserving benchmark of 240 identity-disjoint
2WikiMultiHopQA and MuSiQue questions, two frozen query-state model families,
a generated-subquestion baseline, and a fresh matched four-document retrieval
endpoint. Build and run it from the repository root:

```powershell
python experiments/paper2_7_query_graph/build_natural_facet_benchmark.py
python experiments/paper2_7_query_graph/run_llm_decomposition.py `
  --model gemma3:1b --num-predict 96 `
  --output docs/papers/shared/results/paper2_7_query_graph/natural_facets/llm_decomposition.json

python experiments/paper2_7_query_graph/run_natural_facet_validation.py `
  --model-id Qwen/Qwen3-0.6B `
  --revision c1899de289a04d12100db370d81485cdf75e47ca `
  --llm-predictions docs/papers/shared/results/paper2_7_query_graph/natural_facets/llm_decomposition.json `
  --output-dir docs/papers/shared/results/paper2_7_query_graph/natural_facets/qwen

python experiments/paper2_7_query_graph/run_natural_facet_validation.py `
  --model-id HuggingFaceTB/SmolLM2-135M `
  --llm-predictions docs/papers/shared/results/paper2_7_query_graph/natural_facets/llm_decomposition.json `
  --output-dir docs/papers/shared/results/paper2_7_query_graph/natural_facets/smollm

python experiments/paper2_7_query_graph/run_fresh_retrieval.py `
  --query-features docs/papers/shared/results/paper2_7_query_graph/natural_facets/qwen/natural_query_features.pt `
  --graph-findings docs/papers/shared/results/paper2_7_query_graph/natural_facets/qwen/natural_facet_findings.json `
  --llm-predictions docs/papers/shared/results/paper2_7_query_graph/natural_facets/llm_decomposition.json `
  --output-dir docs/papers/shared/results/paper2_7_query_graph/fresh_retrieval/qwen

python experiments/paper2_7_query_graph/run_fresh_retrieval.py `
  --query-features docs/papers/shared/results/paper2_7_query_graph/natural_facets/smollm/natural_query_features.pt `
  --graph-findings docs/papers/shared/results/paper2_7_query_graph/natural_facets/smollm/natural_facet_findings.json `
  --llm-predictions docs/papers/shared/results/paper2_7_query_graph/natural_facets/llm_decomposition.json `
  --output-dir docs/papers/shared/results/paper2_7_query_graph/fresh_retrieval/smollm

python experiments/paper2_7_query_graph/summarize_next_iteration.py
```

`natural_query_features.pt` files are regenerable caches and are ignored. The
tracked CSV/JSON artifacts preserve per-example metrics, validation choices,
paired intervals, raw LLM responses, parse notes, selections, and gate status.
Generated subquestions are aligned to source words only for direct partition
metrics; retrieval encodes their generated strings directly.
