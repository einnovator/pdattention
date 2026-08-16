# Paper 3.5 adaptive PRA study

Run from the repository root:

```powershell
python -m experiments.paper3_5_adaptive_pra.run_study
```

The adaptive controller is selected on the inherited Paper 2.5 validation
partition and evaluated on its frozen test partition. Output-entropy calibration
uses the separate Paper 3 controlled-model validation/held-out split. Systems
benchmarks are standalone CPU prototype measurements; RAG, long-context, and
KV-cache comparisons are explicitly marked controlled proxies. No full PRA
backbone training is performed.

The study additionally emits the query-region and router-architecture artifacts
specified by the Paper 3.5 add-ons. The query-region gate crosses five layouts,
three payload types, explicit/structural/retry policies, and a 0--8K displacement
sweep. The router gate compares R0 profiles, R1 feature heads, R2 semantic-input
heads, and R3A autoregressive heads under one validation-derived minimum-effort
target. Complexity escalation stops when held-out quality/cost does not improve.
