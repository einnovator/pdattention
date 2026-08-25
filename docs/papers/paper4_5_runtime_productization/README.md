# PRA runtime productization

Paper 4.5 asks whether PRA's logical K/V sparsity becomes a measurable,
portable inference primitive before a serving engine is redesigned around it.

The implementation adds:

- one `PRARuntime` facade over the existing HF model API;
- Paper 4 authenticated cold/warm/hot memory sessions;
- Paper 6.5 typed resource discovery and safe execution;
- versioned runtime configuration and inspection;
- deduplicated interval planning and native `[B, Hkv, T, D]` K/V packing;
- byte-bounded LRU accounting and request-stage profiling;
- eager and `torch.compile` gather gates;
- a scheduler-unaware vLLM handoff contract;
- CLI and executed notebook workflows.

Reproduce the measured portable profile:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_5_runtime.run_runtime_profile
python -m experiments.paper4_5_runtime.summarize_runtime
```

Build the paper:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Try the SDK notebook under `pra-hf-demo/pra_runtime_productization.ipynb`.
