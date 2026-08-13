# Paper 2.5: Iterative PRA

Build from this directory with:

```powershell
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Primary experiment:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_closure.py --device cpu
python ../../../experiments/paper2_5_iterative_pra/summarize_results.py
```

The compact routing sweep is faster on CPU for these small matrices.  The separate
`run_downstream_smoke.py` runner validates full native-K/V execution on CUDA.
