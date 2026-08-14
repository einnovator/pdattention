# Monotonic root and adaptive competition gate

Generate the additive artifacts with:

```powershell
python experiments/paper2_5_iterative_pra/run_monotonic_adaptive_competition.py --device cuda
```

The runner replays the frozen layer-27 feature cache and five learned routing
projections. It does not train a model, change the SDK default, consult evidence
labels while routing, or materialize K/V during search. Root and transition
policies are selected only on the hash-stable validation examples; the paper
reports the held-out split.

The two frozen policy roles are:

- `preservation`: root `z > -0.5`, which locks all available Top-B identities
  on this sample and exactly reproduces one-shot quality;
- `exploration`: native-rank root `z > 1` with fixed `k=1`, and semantic
  seed-agreement `>= 0.6` with fixed `k=4`.

At the held-out 20% budget, native exploration raises HotpotQA exact-oracle
recall from 0.205 to 0.233 but leaves chain completion at 0.143. QASPER recall
and chain completion remain equal to one-shot at 0.883 and 0.800. Across all
held-out exploratory budgets, the first Hotpot evidence group is absent from
full root Top-B in 52.9% of runs and present-but-not-locked in 11.1%. The
artifact therefore recommends root discovery/query decomposition, not another
transition-width sweep.

Important files:

- `adaptive_competition_results.json`: configuration, selection audit, held-out
  summaries, failure decomposition, synthetic controls, and recommendation;
- `root_policy_rows.csv`: full root Top-B, score, lock, final identity, and cost
  traces for the root-policy matrix;
- `transition_policy_rows.csv`: corresponding fixed/adaptive transition matrix;
- `heldout_policy_rows.csv`: per-example/per-seed rows for the four frozen
  preservation/exploration policies;
- `heldout_effects_vs_one_shot.csv`: quality deltas and gain per extra search
  comparison;
- `synthetic_controls.csv`: direct, ambiguous rank-2, and distractor-heavy
  rank-4 controls under matched final budgets;
- `adaptive_quality.{pdf,png}` and `adaptive_regulation.{pdf,png}`: paper plots.

The 30 MB and 27 MB policy-matrix row files intentionally preserve raw score
and rank traces. They are additive and do not replace prior Gate 0-3 or oracle
diagnostic artifacts.
