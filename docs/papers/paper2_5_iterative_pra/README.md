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

Projection-correct and hierarchical-local gates:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_local_associative_closure.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/precompute_local_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_gate2_local_closure.py --device cuda
```

Gate 2 encodes 256-token contextual parents once and derives eight 32-token
local means without re-encoding subwindows. Results and schema-v2 graph traces
are under `../shared/results/paper2_5_iterative_pra/local_associative_closure/`.
