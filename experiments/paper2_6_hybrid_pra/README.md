# Paper 2.6 experiments

The runners in this directory study discovery only. They do not materialize
native K/V and do not generate answers.

## Final iteration

Replay the frozen 132-identity cohort, export detailed confidence provenance,
run controlled ambiguity fixtures, estimate bootstrap stability, and create the
paper artifacts:

```powershell
$env:PYTHONPATH = 'src;.'
python experiments/paper2_6_hybrid_pra/run_final_iteration.py --local-files-only
```

The runner uses the original cohort seed (`20260811`) only to reconstruct the
cached QASPER/Hotpot identities. Bootstrap and calibration diagnostics use the
independent deterministic seed `20260822`.

After changing only aggregation or plots, reuse the detailed candidate rows:

```powershell
python experiments/paper2_6_hybrid_pra/run_final_iteration.py `
  --local-files-only --reuse-detailed
```

Use `--postprocess-only` only when the frozen tensor bundles are unavailable;
that fallback is intentionally limited to confidence fields already serialized
by the preceding channel-selection run.

Outputs are written to
`docs/papers/shared/results/paper2_6_hybrid_pra/final_iteration/`. The bootstrap
rows are uncertainty estimates over the frozen identities, not cohort expansion.
## Normalized PRA efficiency

After rerunning the frozen channel-selection replay, build the normalized
root/successor analysis with:

```powershell
python -m experiments.paper2_6_hybrid_pra.normalized_efficiency
```

The analysis deduplicates selected identities across stages and keeps search
comparisons separate from the conceptual working-set fraction.
