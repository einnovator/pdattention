# Paper 0 claim ledger

Frozen for the 2026-09-12 evidence-first revision. Paper 0 is a position article;
the measurements below are disclosed companion evidence, not independent experiments
performed for this manuscript.

| Claim | Source section | Exact source | Cohort / baseline | Uncertainty unit | Supported wording | Unresolved action |
|---|---|---|---|---|---|---|
| Contextual native K/V can be admitted under a bounded payload while logical scope grows. | Current Evidence and Its Boundaries | `docs/papers/shared/results/recall_sparsity/paper1/fixed_k_vs_fraction.csv`; `docs/papers/shared/results/pra_context_budget.json` | Controlled HotpotQA/QASPER-derived answer-code probes; dense/full and tail controls | Five paired model seeds over frozen held-out examples | Encode-once slicing preserves controlled transport through the split-256 condition while fixed-k active fraction falls to about 5%. | Run matched 32K--8M scaling with source input, direct input, retained memory, and combined attention K/V reported separately. |
| Position handling is part of the native-state contract. | Conceptual Commitments; Current Evidence | `docs/papers/shared/results/paper1_5_rope/rope_translation.json` at evidence commit `2b50f34cbf9f42f5dec596f26493f8d67cfaf43d`; `docs/papers/shared/results/paper1_5_rope/rope_distance_policy.json` at evidence commit `d36f6775270125cd198543b77dc114e6018ee67a` | Frozen positional controls within tested model/backend cells | Paired examples/seeds as declared by the companion | A controlled attention operator is invariant to common translation when relative geometry is unchanged; reset and distance policies show scoped, policy-dependent recovery. No universal policy follows. | Frozen cross-family replication with identical selected identities. |
| Discovery, admission, and consumption can fail independently. | Current Evidence; Risks and Limitations | Companion commit `e81d4345`, `docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_depth_results.json`, `output_validation/gate3_generation_results.json`, and `final_metrics/final_metrics_results.json` | Controlled associative-memory and answer-generation conditions | Unique held-out questions, not rows created by grids or seeds | Traversal can improve while answer utility is flat or worse; this is a diagnostic separation, not an end-to-end benefit. | Learned entry plus consumption on an untouched cohort. |
| Logical token selection is not physical reuse or a speedup. | Implementation Invariants; Virtual Memory; Conclusion | `docs/papers/shared/results/pra_kv_residency.json`; runtime companion commit `d9e6af2d` | Scoped CPU/GPU and resident-cache mechanism tests | Request/session and hardware configuration | Selected-history re-encoding, K/V copies, resident bytes, temporary bytes, and latency must be reported separately. | Concurrent native serving with measured reuse, eviction, cancellation, and matched output trajectories. |

## Claim boundary

The article does not claim production speed, universal pretrained compatibility, general
multi-hop retrieval, or superiority to optimized long context, retrieval, or prefix
caching. A result only supports the contract, model, backend, task, budget, and statistical
unit named by its artifact.
