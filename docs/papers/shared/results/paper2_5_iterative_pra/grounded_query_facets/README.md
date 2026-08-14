# Facet types and query-grounded propagation

This directory contains the additive Gate A and Gate B artifacts for the
facet-type, long-prompt support, and query-grounded propagation study. All
measurements use frozen Qwen3-0.6B features and five frozen semantic routing
projections. No model or router training was performed.

Gate A freezes `w=2`, stride 1, max reduction, global plus local facets, and the
latest user-message support boundary using validation identities only. The
held-out 20% parent-budget result is 0.429 HotpotQA root entry and 0.967 QASPER
root entry. The prior global and `w=4`, stride 2 controls reproduce exactly.

Gate B uses layer-27 pre-RoPE native Q/K Top-4 reduction to propose four
successors from an oracle-conditioned correct first evidence group. Static
semantic query facets then rerank or filter only those candidates. Validation
selects all-facet query reranking, but held-out R@1 changes from 0.286 to 0.257,
MRR from 0.583 to 0.500, and R@4 remains 1.000. The predeclared conditional
success gate therefore fails and no end-to-end propagation run is performed.

Key files:

- `grounded_facet_gate_results.json`: frozen Gate A selection and baseline checks.
- `facet_type_*.csv`: per-run and aggregate facet-family measurements.
- `query_support_*.csv`: clean prompt support-region measurements.
- `stale_contamination_*.csv`: controlled stale-history measurements.
- `grounded_query_feature_manifest.json`: provenance for the ignored feature cache.
- `grounded_propagation_results.json`: frozen Gate B selection and stop decision.
- `grounded_propagation_*.csv`: candidate provenance, metrics, and costs.
- `facet_type_gate.*`, `stale_support_gate.*`, and
  `grounded_propagation_gate.*`: paper figures in PDF and PNG form.

Regenerate from the repository root with the three commands in
`experiments/paper2_5_iterative_pra/README.md`. The large
`grounded_query_features.pt` file is intentionally ignored by Git.
