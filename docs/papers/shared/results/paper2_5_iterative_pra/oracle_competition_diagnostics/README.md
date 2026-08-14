# Oracle competition diagnostics

Generate this additive diagnostic with:

```powershell
python experiments/paper2_5_iterative_pra/run_displacement_calibration.py --device cuda
```

The run replays the five frozen routing methods over five seeds and the canonical
5/10/20/30/40% budgets. It imports the exact Paper-2 annotated-evidence oracle
and requires identity parity with all 4,000 rows in the preceding
`oracle_convergence` artifact. It does not train a router, change SDK routing,
run generation, or materialize K/V during diagnosis.

The new analyses are:

- `displacement_rows.csv`: 2,488 one-shot oracle-hit preservation records.
- `protected_root_rows.csv`: matched-budget, oracle-labeled upper bounds.
- `score_family_rows.csv`: 17,710 full root/propagation score records at 20%.
- `native_sigmoid_saturation.csv`: raw-QK to sigmoid compression statistics.
- `calibration_candidate_rows.csv`: frozen selected-candidate unions.
- `calibration_control_rows.csv`: held-out actual/z-score/quantile controls.
- `convergence_aggregate.csv`: quality, payload, comparisons, dots, and wall time.
- `oracle_recall_cost.{pdf,png}`: active-KV and measured-cost convergence.

At 20%, QASPER one-shot finds some oracle evidence in 93.8% of runs. Native max
and Top-4 displace 11.7% and 15.6% of those oracle-parent hits. Protecting them
within the same final budget raises native recall from 0.794/0.770 to 0.838.
Every native candidate sigmoid exceeds 0.999, but validation-only family
normalization gives smaller held-out gains and no Hotpot recovery advantage.
The artifact therefore recommends `protected_root` as the next isolated gate.
