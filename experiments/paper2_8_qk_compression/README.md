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

## Query-conditioned continuation

Run the low-rank query-conditioned matrix after the gated key-only study:

```powershell
python experiments/paper2_8_qk_compression/run_query_conditioned_study.py `
  --device cuda
python experiments/paper2_8_qk_compression/summarize_query_conditioned.py
```

The matrix crosses ranks `8,16,32`, landmark counts `4,8`, four objectives,
and five seeds, for 120 trained controllers. `combined/r16/m4` is fixed as the
primary configuration before test evaluation. All other best-of-matrix rows
are exploratory. The runner is resumable and writes row-level results after
each configuration. Its small prepared-feature cache is regenerable and
ignored; controller checkpoints, histories, comparisons, and plots are kept
under `query_conditioned/`.

The primary controller reaches HotpotQA recall `0.1667` and QASPER recall
`0.0574`. Exploratory decision-aware `r32/m8` reaches QASPER recall `0.1718`,
near exact routing at `0.1776`, but its paired intervals include zero. The best
response-imitation controller recovers `47.3%` of greedy-oracle preservation
gain, still below the prespecified `80%` G3 threshold.

## Post-G3 low-rank frontier

The diagnostic extension preserves G0-G3 and adds matched native geometry,
matched selector ablations, and direct low-rank routing:

```powershell
python experiments/paper2_8_qk_compression/run_low_rank_frontier.py --device cuda
python experiments/paper2_8_qk_compression/run_selector_ablation.py --device cuda
python experiments/paper2_8_qk_compression/summarize_low_rank_frontier.py
```

The direct router projects flattened native keys (`1024 -> r`) and native
queries (`2048 -> r`) at ranks `8,16,32`. The projected token index is cached;
online routing performs only the query projection and low-rank dots. Static
k-means, medoid, and farthest-first indexes then test joint token and feature
compression at four and eight representatives. These routing summaries never
replace the selected chunks' native K/V during materialization.

`low_rank_frontier/` keeps routing-index bytes separate from backing native-K/V
bytes, selected-K/V transfer, active materialization, construction time, cached
online latency, native dots, low-rank dots, and eager GPU peak deltas. The
extension is diagnostic: its E0-E6 decisions do not revise the original failed
G3 or open recurrent/synthetic memory.

On the frozen cohort, the all-token rank-16 router reaches QASPER evidence
recall `0.2542` with `2048` routing-index bytes per chunk. Rank 32 is the best
direct HotpotQA row at `0.1454`, still far below lexical controls. Joint
compression with rank-8, eight-centroid indexes reaches QASPER recall `0.1829`
using `256` bytes per chunk. These are evidence-supervised routing results, not
faithful teacher compression: teacher top-four overlap remains low. E2 is
inconclusive, so the extension does not expand to new datasets.
