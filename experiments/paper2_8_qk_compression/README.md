# Paper 2.8 experiments

Paper 2.8 tests whether a small native-key subset preserves the chunk ranking
induced by a frozen transformer's full token-level key stream. Candidate memory
units remain the inherited 32-token chunks, and every routing method requests
the same four chunks for downstream native K/V materialization.

Generate the identity-disjoint validation cache with the same Qwen3-0.6B
revision and layer-27 capture used by Paper 2.5:

```powershell
python experiments/paper2_5_iterative_pra/precompute_native_qk_features.py `
  --device cuda --split validation --offset 0 --examples 8 `
  --output-dir docs/papers/shared/results/paper2_8_qk_compression
```

The test feature file is the untracked artifact named by Paper 2.5's native-QK
manifest. Place or hard-link it at
`docs/papers/shared/results/paper2_8_qk_compression/native_qk_features_test.pt`.

Run the gates in order:

```powershell
python experiments/paper2_8_qk_compression/run_gated_study.py --device cuda
```

The runner selects the teacher function on validation identities, evaluates
full-K, mean, last, random, farthest-first, and greedy-oracle controllers, and
opens five-seed learned-selector training only if G2 passes. The tracked output
directory contains row-level CSVs, paired bootstrap effects, changed-selection
audits, plots, gate decisions, tiny selector checkpoints when trained, and a
reproducibility manifest. Large regenerable Q/K feature tensors are ignored.

Generate the uncached synchronized controller-cost audit, then rebuild the
matched Paper 2.5/2.6 tables and figures:

```powershell
python experiments/paper2_8_qk_compression/run_cost_benchmark.py --device cuda
python experiments/paper2_8_qk_compression/summarize_results.py
```

`natural_rows.csv` records cached compact-key assembly time because controller
indices are reused across `m`. Use `controller_cost_rows.csv` and
`controller_cost_summary.csv` for end-to-end controller construction plus QK
scoring latency.
