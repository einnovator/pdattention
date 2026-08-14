# Query-facet and native-head entry discovery

This additive diagnostic tests whether HotpotQA root-entry failures come from
collapsing a contextual question across token spans or native query heads too
early. The frozen Qwen3-0.6B backbone, five learned semantic-router seeds,
memory gists, final parent budget, and native K/V payload remain unchanged.
No model or router is trained.

The query is encoded once in its complete prompt. The exact old final-token
state is retained as the global facet; overlapping local facets are means over
states from that same contextual pass. Validation selects a 4-token window,
stride 2, and max reduction. It raises held-out Hotpot first-evidence presence
at the 20% parent budget from 0.343 to 0.457, MRR from 0.503 to 0.626, and
oracle recall from 0.205 to 0.295. QASPER root presence remains 0.967 and oracle
recall remains 0.883. The paired Hotpot changes comprise six gains, two losses,
and 27 ties across seven identities and five seeds.

The learned 128-D semantic projection has no legitimate head axis. Native-head
conditions therefore use real layer-27 pre-RoPE Q heads and their exact 16:8
GQA mapping, with native mean-head controls separating representation from head
reduction. They are a head-specialization audit, not pseudoheads grafted onto
the semantic router. Global native split-head search reaches Hotpot root
presence 0.429 but reduces QASPER to 0.667; combining facets and split heads is
worse. The supported interpretation is query-span dilution, not head dilution.

Because the root gain exceeded the predeclared 0.10 threshold with no QASPER
loss, a small confirmation reconnects the selected facet root to the already
frozen monotonic `zscore_1` plus native-rank fixed-Top-1 propagation policy. At
20%, facets raise Hotpot chain completion from 0.143 to 0.171 and exact-oracle
completion from 0.000 to 0.029. Propagation preserves chain completion at 0.171
and raises exact-oracle completion to 0.057, but requires about 112,232 routing
comparisons versus 68 for root-only facets. QASPER remains at 0.800 chain
completion under all four comparison methods. Query facets are therefore an
opt-in research result; they do not change SDK defaults.

Regenerate the artifacts from the repository root:

```powershell
python -m experiments.paper2_5_iterative_pra.precompute_query_entry_features --device cuda
python -m experiments.paper2_5_iterative_pra.run_query_entry_facets --device cuda
python -m experiments.paper2_5_iterative_pra.run_query_entry_propagation --device cuda
```

`query_entry_features.pt` is regenerable and ignored. Its tracked manifest pins
the model revision, artifact hash, tensor topology, and one-pass invariant.
Tracked JSON/CSV files retain selection audits, per-example scores, span/head
provenance, matched budgets, synthetic controls, and the propagation comparison.
