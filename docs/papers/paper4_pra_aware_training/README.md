# PRA-aware training

Paper 4 now has three explicit stages:

1. The completed five-seed controlled consumer-learning gate under fixed oracle
   native K/V.
2. A resolver-neutral cold/warm/hot external-memory lifecycle mechanism study.
3. An exact-native-global-slot Gemma 3 1B adaptation gate whose long training
   runs are intentionally pending distributed or larger-memory configuration.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Regenerate tables, plots, and result macros first:

```powershell
python -m experiments.paper4_training.summarize_tier0
```

Regenerate the external-memory lifecycle artifacts with:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_training.run_external_memory_lifecycle
```

Regenerate the Gemma architecture and compute gate from the pinned local config:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_training.prepare_gemma_gate `
  --config D:\huggingface-cache\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752
```

Do not launch G2--G5 until a 100--500-step benchmark has been run on the final
distributed/larger-memory setup and its wall-clock and memory budget approved.
