# PRA-aware training

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Regenerate tables, plots, and result macros first:

```powershell
python -m experiments.paper4_training.summarize_tier0
```

The pretrained gate covers SmolLM2 135M plus five-seed Gemma 3 270M and 1B
Consumer/Interface adaptation. All scopes improve generic NLL, but none clears
the all-seed relevant-versus-distractor criterion. Learned routing therefore
remains disabled. Reproduce or resume the Gemma grid with:

```powershell
python -m experiments.paper4_training.run_gemma_adaptation_grid
```
