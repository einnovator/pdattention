# Toy-model causal diagnosis

This audit ranks H1-H9 only after the frozen v6 causal captures were generated.

| Hypothesis | Status | Artifact-backed observation |
|---|---|---|
| H1 retrieval failure | supported | Selected iterative complete-path recovery is 0.408, versus 1.000 under a matched oracle-forced plan. |
| H2 weak memory attention | unsupported as a general explanation | Oracle evidence receives 0.270 final-query attention mass; the memory path is active. |
| H3 softmax dilution | partially supported | Selected evidence receives 0.147 mass versus 0.521 for memory distractors, and the oracle raises margin by +1.552. Shuffling remains a caution against attributing every selected-memory gain to evidence content. |
| H4 wrong-direction update | supported for bad memory, not oracle memory | Wrong memory lowers accuracy to 0.057; oracle raises it from 0.140 to 0.398. Alignment signs track this ordering. |
| H5 later-layer erasure | partially supported, secondary | 21.6% of oracle traces with a positive immediate margin effect lose it by the final layer. |
| H6 intervention-density interference | partially supported | Recovery and K/V state count continue to move after answer accuracy becomes non-monotonic; see `intervention_density_frontier.csv`. |
| H7 representation-depth mismatch | supported | Oracle usefulness is strongest in early consumer layers and degrades toward layer 5. |
| H8 frozen-consumer mismatch | unsupported as an absolute bottleneck; adaptation benefit unresolved | Frozen oracle memory raises accuracy by +0.258, proving that the unadapted consumer can use correctly placed evidence. |
| H9 task saturation | unsupported as the primary explanation | When iterative path recovery improves, margin changes by +2.164 and accuracy by +0.102. |

The dominant classification is **B1 retrieval-to-attention**, qualified by
**B4 intervention scheduling** and **B3 partial later erasure**. Oracle evidence
is attended and used productively, so B2/B5 are not fundamental incapacity
claims. The remaining design problem is to select evidence precisely and place
memory interventions where the frozen residual stream can preserve their effect.
