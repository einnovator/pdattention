# Paper 7 typed adaptive-context experiments

`run_inception_experiments.py` runs the bounded mechanism study used by the
inception paper. It covers five seeds and keeps model-dependent claims out of
the first implementation checkpoint:

- type-specific compact-view savings;
- exact backing-state recovery;
- compact, explicit-address, latent-query, and proactive-probe trigger reachability;
- native-event, tool, mixed-cursor, and proactive materialization accounting;
- cursor aggregate/drill-down correctness;
- adaptive transport decisions across payload sizes and deployment topologies.

Run from the repository root:

```powershell
python experiments/paper7_records/run_inception_experiments.py
```

Artifacts are written under
`docs/papers/shared/results/paper7_records/inception/`.
