# Paper 2.5 iterative PRA experiments

Run commands from the repository root.

## Projection and local-associative gates

Gate 1 corrects asymmetric frontier projection without retraining:

```powershell
python experiments/paper2_5_iterative_pra/run_local_associative_closure.py --device cuda
```

Gate 2 captures 256-token contextual parents with eight 32-token local means,
then compares one-shot parent routing, projection-correct parent closure, and
local-gist closure under the same final parent/KV budget:

```powershell
python experiments/paper2_5_iterative_pra/precompute_local_features.py --device cuda
python experiments/paper2_5_iterative_pra/run_gate2_local_closure.py --device cuda
```

Artifacts are written under
`docs/papers/shared/results/paper2_5_iterative_pra/local_associative_closure/`.
The public SDK keeps one-shot routing as its default. The opt-in
`routing_mode="local_iterative"` path requires a routing adapter.

## Query-facet root discovery

The additive entry-discovery study captures the complete contextual query once,
derives overlapping local facets without independent window re-encoding, and
compares them with the exact final-token baseline and real pre-RoPE native-head
controls. If the validation-selected facet condition clears the root-gain gate,
the third command runs the predeclared bounded-propagation confirmation.

```powershell
python -m experiments.paper2_5_iterative_pra.precompute_query_entry_features --device cuda
python -m experiments.paper2_5_iterative_pra.run_query_entry_facets --device cuda
python -m experiments.paper2_5_iterative_pra.run_query_entry_propagation --device cuda
```

Artifacts are written under
`docs/papers/shared/results/paper2_5_iterative_pra/query_entry_facets/`.

## Gate 3: native local Q/K closure

Gate 3 preserves the frozen semantic router for initial relevance and candidate
narrowing, then compares tokenwise layer-27 pre-RoPE Q/K inside contextual
32-token regions. It does not load a memory-use adapter or change final post-RoPE
native K/V materialization.

```powershell
python experiments/paper2_5_iterative_pra/precompute_native_qk_features.py --device cuda
python experiments/paper2_5_iterative_pra/run_gate3_native_qk_closure.py --device cuda
```

The first command writes a regenerable 621 MB tensor cache under
`docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure/`. The cache
is intentionally ignored; its pinned-model provenance, byte count, SHA-256,
results, graphs, and plots are tracked. The predeclared run uses five router
seeds, final parent budgets of 10/20/30%, semantic candidate pools of 10/20%,
max and Top-4 reductions, and one `mu + sigma` threshold control.
