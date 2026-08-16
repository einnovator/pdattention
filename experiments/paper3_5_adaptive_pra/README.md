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
