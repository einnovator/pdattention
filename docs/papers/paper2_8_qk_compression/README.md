# Paper 2.8: QK-response distilled memory landmarks

Paper 2.8 tests whether a small subset of native keys can preserve the frozen
transformer's full-key chunk ranking better than one mean key. The study uses
the frozen Paper 2.5/2.6 Qwen3-0.6B cohort, 32-token candidate chunks, a matched
four-chunk materialization budget, and five learned-selector seeds.

The gate sequence stops at G3. A query-aware greedy oracle establishes
response-preservation and QASPER retrieval headroom, but the 321-parameter
key-only selector recovers 40.5% of oracle gain rather than the required 80%.
A 120-controller continuation then crosses low-rank query conditioning,
landmark count, and four training losses. Prespecified combined `r16/m4`
improves HotpotQA recall to 0.1667 but reduces QASPER recall to 0.0574.
Exploratory decision-aware `r32/m8` reaches 0.1718 on QASPER, near exact
routing at 0.1776, with paired intervals that include zero. Maximum oracle-gain
recovery rises only to 47.3%. Synthetic slots and streaming recurrent memory
are therefore not run.

A separately labeled post-G3 diagnostic tests native geometric controls and
direct low-rank native-QK routing without changing materialization. All-token
rank 16 reaches QASPER evidence recall 0.2542 with 2,048 index bytes per chunk;
rank-8, eight-centroid routing reaches 0.1829 with 256 bytes. HotpotQA remains
dominated by lexical routing. Because the projections have low teacher overlap
and use validation evidence supervision, the paper interprets them as compact
task-adapted routing features rather than faithful native-response compressors.

Files:

- `paper_2_8.tex` and `paper_2_8.pdf`: measured manuscript and built paper.
- `AGENTS.md`: original gated research contract.
- `../../../experiments/paper2_8_qk_compression/`: runners and reproduction
  commands.
- `../shared/results/paper2_8_qk_compression/`: row-level metrics, bootstrap
  effects, changed-selection audits, plots, selector checkpoints, costs, gates,
  and manifests.

The `query_conditioned/` result subtree contains all 120 controller runs,
training histories, seed-stability summaries, paired effects, response
recovery, and publication plots. The test-selected QASPER configuration is
marked exploratory throughout the paper and artifacts.

The `low_rank_frontier/` and `selector_ablation/` subtrees contain the direct
projection and joint-compression sweeps, five-seed structural ablations,
identity-paired bootstrap effects, cost-frontier tables, extension gates,
checkpoints, and publication plots.

The large validation/test QK feature tensors are reproducible and intentionally
ignored. Their hashes and generation commands are recorded in `manifest.json`.
