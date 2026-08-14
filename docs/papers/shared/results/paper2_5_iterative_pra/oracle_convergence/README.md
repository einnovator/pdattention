# Oracle convergence and true-edge rank

This directory is additive to Gates 1--3. Generate it with:

```powershell
python experiments/paper2_5_iterative_pra/run_oracle_convergence.py --device cuda
```

The run uses the frozen five projection seeds, exact Paper-2 annotated-evidence
oracle, 5/10/20/30/40% parent budgets, and the existing one-shot, parent/local
semantic, native-max, and native-Top-4-reduction routers. No downstream model is
run and no routing representation is trained.

Key files:

- `oracle_convergence_rows.csv`: all 4,000 per-example/seed/budget selections.
- `oracle_convergence_aggregate.csv`: dataset/method budget curves.
- `oracle_edge_rank_rows.csv`: parent, local, native-max, and native-Top-4 scores.
- `oracle_edge_rank_summary.csv`: rank quartiles, R@K, MRR, and margins.
- `geometry_comparison_rows.csv`: paired semantic/native rank classifications.
- `top4_signal_cases.csv`: exact Gate-3 Top-4 wins over max reduction.
- `adaptive_competition.csv`: validation-selected, held-out offline policy results.
- `oracle_convergence_results.json`: compact metadata and decision summary.

At 20% Hotpot budget, native Top-4 reduction has the highest oracle recall
(0.265) but only 0.013 complete recovery. Conditioned on the first evidence
group, all three primary geometries have median edge rank 3 and R@4 of
0.741--0.765. The result recommends bounded competition as the next diagnostic,
while the tested Top-1/Top-2 margin threshold is not yet an effective regulator.
