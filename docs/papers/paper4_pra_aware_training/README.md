# PRA-aware training

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Regenerate tables, plots, and result macros first:

```powershell
python -m experiments.paper4_training.summarize_tier0
```
