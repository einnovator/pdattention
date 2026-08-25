# Paper 4.5: PRA runtime productization

## Claim boundary

This branch unifies the Paper 2 model API, Paper 4 external-memory lifecycle,
and Paper 6.5 typed resource/execution boundary. The supplied runtime brief was
headed "Paper 5.5"; the branch and paper use the user-requested Paper 4.5
identifier without changing its systems scope.

The current evidence supports a portable eager materialization baseline and a
measurement contract. It does not support a `torch.compile`, Triton, custom
CUDA, vLLM, SGLang, TensorRT-LLM, or MLX speed claim. Optional runtime paths
must be reported as measured, contract-only, unavailable, or architectural.

Do not infer physical HBM savings from selected-token fractions. Report warm
resident detail K/V, hot selected K/V, transfer bytes, temporary packing bytes,
and peak accelerator allocation separately.

## Reproducibility

Run the runtime profile and summary from the repository root:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_5_runtime.run_runtime_profile
python -m experiments.paper4_5_runtime.summarize_runtime
```

Build `paper.tex` with `latexmk`. Private checkpoints, raw model states, and
credentials are not paper artifacts.
