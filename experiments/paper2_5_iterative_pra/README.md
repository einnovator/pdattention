# Paper 2.5 iterative PRA experiments

Run commands from the repository root.

## Controlled LocalSA reopening

The causal receptive-field experiment is self-contained. It does not depend on
another paper's materialization policy. Each randomized chain uses balanced
terminal labels, several label-relation decoys, reverse-causal evidence order,
and random distractor interleaving. All windows and initialization seeds share
the exact same generated train/evaluation examples.

```powershell
python -m experiments.paper2_5_iterative_pra.run_controlled_local_sa `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6 `
  --device cuda --windows 16,32,64,128,global --seeds 17,29,41,53,67 `
  --steps 800 --d-model 96 --layers 6
python -m experiments.paper2_5_iterative_pra.run_controlled_pra `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6 `
  --device cuda --windows 16,32,64,128,global --seeds 17,29,41,53,67 `
  --examples 32 --d-model 96 --layers 6
python -m experiments.paper2_5_iterative_pra.summarize_controlled_local_sa `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6
python -m experiments.paper2_5_iterative_pra.run_toy_mechanistic `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6 `
  --device cuda --windows 16,32,64,128,global --seeds 17,29,41,53,67 `
  --examples 16
python -m experiments.paper2_5_iterative_pra.summarize_outcome_b `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6
```

The selected-chunk payload is one indivisible five-token fact. The whole-parent
control is therefore the same fact payload, and the fixed native K/V budget is
20 tokens. One-shot routes up to four facts once; progressive conditions route
one unseen fact per intervention and count layer-token K/V states. Gold URI
paths are used only after inference for scoring.

The mechanistic pass reuses the validation-selected checkpoints and cached
layer-native projections. It compares no memory, routed memory, oracle-only
evidence, matched irrelevant memory, and content-shuffled memory. Forward hooks
capture final-head margins after each PRA residual and native layer, plus exact
evidence/distractor/native shared-softmax mass. No model parameter is updated.

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

## Facet-type and query-grounded propagation gates

This additive iteration keeps the source memories, learned projections, native
transition geometry, and final parent budget fixed. Gate A compares fixed,
token, phrase, multiscale, global/local, and robust-reduction facet families.
It then captures controlled chat prompts containing stale memory-matching
history and freezes one query-support boundary on validation identities. Gate B
conditions on a correct first evidence group: native Q/K Top-4 proposes a
bounded successor set, and the frozen query facets validate only those
identities. Native and semantic raw scores are never added.

```powershell
python -m experiments.paper2_5_iterative_pra.precompute_grounded_query_features --device cuda
python -m experiments.paper2_5_iterative_pra.run_grounded_facet_gate --device cuda
python -m experiments.paper2_5_iterative_pra.run_grounded_propagation_gate --device cuda
```

The first command writes an ignored, regenerable contextual-state cache. The
tracked manifest, raw rows, validation audits, plots, and result JSON are under
`docs/papers/shared/results/paper2_5_iterative_pra/grounded_query_facets/`.
The conditional propagation runner executes an end-to-end comparison only when
held-out conditional R@1 improves by at least 0.10 while losing at most 0.05
R@4. The present run stops at that gate and does not change the SDK default.

## Dynamic query reconstruction gate

This additive gate grants the correct first evidence group A, reconstructs a
current state with one frozen Qwen forward over `Q || A`,
`Q || [Retrieved memory] || A`, or `A || Q`, and derives width-2 contextual
facets from either Q alone or Q plus A. The existing layer-27 pre-RoPE native-QK
scorer and all five frozen semantic projections are unchanged. The second
command compares static Q0 and reconstructed Q1 over native candidate breadths
1, 2, 3, 4, 5, 6, 8, and 11.

```powershell
python -m experiments.paper2_5_iterative_pra.precompute_dynamic_query_features --device cuda
python -m experiments.paper2_5_iterative_pra.run_dynamic_query_gate --device cuda
```

The ignored feature cache and tracked manifest, rows, selection audit, bridge
diagnostics, K plot, and result JSON are under
`docs/papers/shared/results/paper2_5_iterative_pra/dynamic_query_discovery/`.
The validation-selected dynamic state fails the predeclared held-out +0.10 R@1
gate, so the larger facet/K/active-budget/threshold surface and dynamic closure
are intentionally not run. This gate performs no generation or native-KV
materialization and does not change the SDK default.

## Terminal-query semantic graph search

This additive diagnostic uses the query only to enter and terminate search.
Intermediate parents are proposed and admitted exclusively by frozen native
local-QK association. Edge and goal thresholds are calibrated independently,
and oracle-root success is required before routed roots can run.

```powershell
python -m experiments.paper2_5_iterative_pra.run_semantic_graph_search --device cuda
python -m experiments.paper2_5_iterative_pra.review_semantic_graph_false_goals
```

CUDA builds the parent graph and profiles the selected condition. The broad
K/B/threshold surface runs from the same CPU score cache to avoid per-condition
GPU synchronization. The oracle-root traversal recovers held-out evidence, but
the terminal predicate fails its false-goal gate; routed roots are therefore
absent by design. Rows, provenance, plots, and the decoded terminal review are
under `docs/papers/shared/results/paper2_5_iterative_pra/semantic_graph_search/`.

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

## Hotpot chunk granularity and oracle discovery

This additive control keeps the frozen native graph-search algorithm fixed,
grants one deterministic first-evidence root, and varies only zero-overlap
parent size, K, H, and B. Evidence labels are attached after search. A small
ignored cache supplies exact layer-27 token hidden states for semantic parent
means; native edges reuse exact cached tokenwise pre-RoPE Q/K.

```powershell
python -m experiments.paper2_5_iterative_pra.precompute_chunk_granularity_features --device cuda
python -m experiments.paper2_5_iterative_pra.run_chunk_granularity --device cuda
```

The runner asserts exact reproduction of canonical 256-token held-out
K4/B6/H4 recovery, emits the complete 16--256-token discovery surface,
computes facet-parent diagnostics over five frozen projections, and validates
chain contraction controls. No native K/V is materialized. Artifacts are under
`docs/papers/shared/results/paper2_5_iterative_pra/chunk_granularity/`.

## Layerwise native graph and contextualization

This experiment holds the 128-token search partition and canonical exact
native-Q/K graph policy fixed while exposing actual Q/K states from decoder
layers 0, 4, 8, 12, 16, 20, 24, and 27. The same forwards capture the native
attention and FFN residual contributions. Position-preserving causal masks then
restrict each token to self, 16-token, or 32-token history without rebinding its
absolute logical position.

```powershell
python -m experiments.paper2_5_iterative_pra.precompute_layerwise_graph_features --device cuda
python -m experiments.paper2_5_iterative_pra.run_layerwise_graph_exploration --device cuda
```

Large per-example Q/K tensors are regenerable and ignored. Their SHA-256
manifest, token-class contextualization summaries, graph rows, plots, and final
result JSON are tracked under
`docs/papers/shared/results/paper2_5_iterative_pra/layerwise_graph/`.

## Final measurement gate

This measurement-only synthesis consumes the frozen natural-graph,
multiscale-query, layerwise, granularity, and systems rows. It does not train a
selector, rerun graph search, generate text, or materialize native K/V.

```powershell
python -m experiments.paper2_5_iterative_pra.run_final_metrics
```

The command evaluates label-free facet diagnostics on disjoint validation and
held-out identities, bootstraps matched-example layer correlations, decomposes
edge and downstream search loss, and emits sparse quality/payload frontiers.
Tables, plots, the negative-results registry, and the strict result manifest are
under
`docs/papers/shared/results/paper2_5_iterative_pra/final_metrics/`.
