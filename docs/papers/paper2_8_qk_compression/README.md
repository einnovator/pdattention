# Paper 2.8: Low-rank native-QK routing

Paper 2.8 studies a compact address index for Progressive Retrieval Attention.
Frozen native queries and keys are projected into a small interaction space;
the resulting rows select chunk identities, while selected chunks still supply
their unchanged original native K/V. The paper therefore measures routing-index
residency separately from backing-memory storage and active materialization.

On 64 fresh QASPER identities, rank-16 routing raises four-chunk evidence recall
from 0.117 for one full-width mean key to 0.258. The architecture transfers to
2WikiMultiHopQA: mean recall is 0.0977, inherited rank-16 reaches 0.1305, and
dataset-specific training reaches 0.1903. A rank-8/eight-centroid index retains
84.9% of the 2Wiki rank-16 gain at about 256 FP32 bytes per chunk. A SmolLM2
replication preserves the QASPER effect across a second model family and
tokenizer.

The channel is deliberately bounded. BM25 remains strongest on HotpotQA,
MuSiQue remains lexical, and validation-selected static fusion regresses on both
new cohorts. Native-K/V causal controls also show that better evidence routing
does not guarantee beneficial use by a frozen late-layer consumer. The paper
positions low-rank native-QK as one compact routing channel, not a universal
retriever, an active K/V compressor, or a complete memory-consumption method.

The original response-distillation, landmark-selection, 120-controller, and
geometry studies remain in the appendices. They document why teacher-response
fidelity was rejected as the organizing objective and why direct
evidence-supervised low-rank interaction became the main mechanism.

Files:

- `paper_2_8.tex` and `paper_2_8.pdf`: measured manuscript and built paper.
- `AGENTS.md`: original gated research contract.
- `../../../experiments/paper2_8_qk_compression/`: runners and reproduction
  commands.
- `../shared/results/paper2_8_qk_compression/`: row-level metrics, bootstrap
  effects, changed-selection audits, plots, selector checkpoints, costs, gates,
  and manifests.

The `confirmation/`, `multi_dataset/`, `low_rank_frontier/`,
`cross_model_smollm2/`, and `confirmation_generation/` result subtrees contain
the main fresh-cohort evidence, paired effects, cost frontier, cross-model
replication, and causal controls.

The `query_conditioned/` subtree retains all 120 historical controller runs,
training histories, seed-stability summaries, response-recovery diagnostics,
and appendix plots. Its test-selected QASPER configuration remains explicitly
exploratory.

The `low_rank_frontier/` and `selector_ablation/` subtrees contain the direct
projection and joint-compression sweeps, five-seed structural ablations,
identity-paired bootstrap effects, cost-frontier tables, extension gates,
checkpoints, and publication plots.

The large validation/test QK feature tensors are reproducible and intentionally
ignored. Their hashes and generation commands are recorded in `manifest.json`.
