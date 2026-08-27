# Paper 8 production-PRA experiment

This experiment holds the Paper 7 native hybrid router fixed and changes the
task partition admitted before routing. It reports selection, typed-record
materialization, evidence availability, and answer consumption separately.

The primary cases cover low, medium, and high semantic confusability across
independent, linear, join, resumption, and conflicting-state scenarios. Task
identity and dependencies are oracle harness metadata; model-produced task
acquisition remains deferred.

## Run

```powershell
$env:PYTHONPATH = "src;."
python experiments/paper8_tasks/run_production_pra.py --phase route
python experiments/paper8_tasks/run_production_pra.py --phase model
python experiments/paper8_tasks/run_production_pra.py --phase postprocess
```

`route` and `model` use CUDA by default and append restartable JSONL
checkpoints. `postprocess` does not load a model. The pinned model is
`Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`.

Generated tables and checkpoints live under
`docs/papers/shared/results/paper8_tasks/production_pra/`; figures live under
`docs/papers/shared/figures/paper8_tasks/`.

The `*_LEXICAL` rows retain the historical lexical/recency control. The main
production rows use Paper 7 encoded native-Q/K hybrid routing. Conditions with
the `_VISIBLE` suffix re-enter selected exact typed records into the visible
prompt to diagnose whether a failure occurred at selection/materialization or
at frozen native-memory consumption.
