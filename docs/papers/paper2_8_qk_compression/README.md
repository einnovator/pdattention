# Paper 2.8: QK-response distilled memory landmarks

Paper 2.8 tests whether a small subset of native keys can preserve the frozen
transformer's full-key chunk ranking better than one mean key. The study uses
the frozen Paper 2.5/2.6 Qwen3-0.6B cohort, 32-token candidate chunks, a matched
four-chunk materialization budget, and five learned-selector seeds.

The gate sequence stops at G3. A query-aware greedy oracle establishes
response-preservation and QASPER retrieval headroom, but the 321-parameter
key-only selector recovers 40.5% of oracle gain rather than the required 80%.
It improves the inherited QASPER semantic-gist baseline but does not beat exact
routing, does not improve Hotpot, and is seed-sensitive. Synthetic slots and
streaming recurrent memory are therefore not run.

Files:

- `paper_2_8.tex` and `paper_2_8.pdf`: measured manuscript and built paper.
- `AGENTS.md`: original gated research contract.
- `../../../experiments/paper2_8_qk_compression/`: runners and reproduction
  commands.
- `../shared/results/paper2_8_qk_compression/`: row-level metrics, bootstrap
  effects, changed-selection audits, plots, selector checkpoints, costs, gates,
  and manifests.

The large validation/test QK feature tensors are reproducible and intentionally
ignored. Their hashes and generation commands are recorded in `manifest.json`.
