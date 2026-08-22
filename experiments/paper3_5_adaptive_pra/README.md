# Paper 3.5 adaptive PRA study

Run from the repository root:

```powershell
python -m experiments.paper3_5_adaptive_pra.run_study
```

The adaptive controller is selected on the inherited Paper 2.5 validation
partition and evaluated on its frozen test partition. Output-entropy calibration
uses the separate Paper 3 controlled-model validation/held-out split. Systems
benchmarks are standalone CPU prototype measurements; RAG, long-context, and
KV-cache comparisons are explicitly marked controlled proxies. No full PRA
backbone training is performed.

The study additionally emits the query-region and router-architecture artifacts
specified by the Paper 3.5 add-ons. The query-region gate crosses five layouts,
three payload types, explicit/structural/retry policies, and a 0--8K displacement
sweep. The router gate compares R0 profiles, R1 feature heads, R2 semantic-input
heads, and R3A autoregressive heads under one validation-derived minimum-effort
target. Complexity escalation stops when held-out quality/cost does not improve.

Run the factorized control and bounded corrective-retry study separately:

```powershell
python -m experiments.paper3_5_adaptive_pra.factorized_study
```

This runner evaluates the independent interpretation/search/admission lattice,
trains R0/R1/R2/R3A on validation-derived factorized targets, and writes compact
oracle, Pareto, precision/recall, router, and retry artifacts. Large frozen
feature tensors remain outside Git. Systems optimization is out of scope for
Paper 3.5 and is handed to Papers 5.5 and 6.

Run the adaptive root/successor search-method study with:

```powershell
python -m experiments.paper3_5_adaptive_pra.adaptive_search_methods
python -m experiments.paper3_5_adaptive_pra.normalized_efficiency
```

The final command reports candidate-normalized search and admission breadth,
evidence-normalized working-set overhead, physical native-K/V tokens, and
bounded-retry increments without collapsing them into one abstract cost.

This runner imports the deterministic Paper 2.6 action specification, replays
its frozen four-dataset channel rows, and learns validation-only root,
successor, useful-address, and targeted-retry selectors. It keeps search
operation counts separate from the inherited Paper 3.5 K/V-admission surface;
their 32-identity join is explicitly a composed diagnostic, not an end-to-end
generation or materialization experiment.

Capture and evaluate the frozen-backbone self-router add-on with:

```powershell
python -m experiments.paper3_5_adaptive_pra.precompute_self_router_features --device cuda
python -m experiments.paper3_5_adaptive_pra.self_router_study --device cuda
```

The capture pools only the explicit question span from the normal question-only
chat prompt. It records embeddings, representative hidden layers, native Q/K,
and a separately charged 256-token source-contextualized upper bound. The study
uses four-fold example-grouped validation selection, five router seeds, and the
existing 792-action frozen surface. It expands target architectures only for the
validation-selected self representation. Held-out maxima never select a layer.

Query-prefill reuse is claimed only at or before the first PRA consumer layer.
Offline tests require exact Qwen prefix-continuation parity and no model-state
mutation. The external-encoder gate remains deferred when self-encoding does not
improve the validation-selected quality/cost frontier.
