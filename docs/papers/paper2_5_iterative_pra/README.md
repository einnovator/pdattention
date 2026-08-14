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

Native local Q/K gate:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_native_qk_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_gate3_native_qk_closure.py --device cuda
```

Gate 3 uses the exact pinned Qwen revision and captures tokenwise layer-27
pre-RoPE Q/K from each single contextual parent encoding. The large regenerable
feature tensor is ignored; its manifest hash and all result/graph/plot artifacts
are under `../shared/results/paper2_5_iterative_pra/native_qk_closure/`.

Oracle convergence and edge-rank diagnostic:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_oracle_convergence.py --device cuda
```

This additive diagnostic imports the Paper-2 annotated-evidence oracle unchanged,
evaluates all five frozen routing methods at 5/10/20/30/40% budgets, and measures
the rank of true Hotpot transitions after forcing only the source evidence group.
It also runs a validation-only offline margin policy; it does not change the SDK
or train a router. Results are under
`../shared/results/paper2_5_iterative_pra/oracle_convergence/`.

Root displacement and score calibration diagnostic:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_displacement_calibration.py --device cuda
```

This runner requires row-for-row identity parity with the oracle-convergence
artifact, classifies one-shot oracle-hit displacement, computes a matched-budget
oracle-protected upper bound, audits root/semantic/native score paths, and tests
validation-only family z-score and empirical-quantile controls on held-out
examples. It remains offline and additive. Results are under
`../shared/results/paper2_5_iterative_pra/oracle_competition_diagnostics/`.

Monotonic root persistence and adaptive competition:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_monotonic_adaptive_competition.py --device cuda
```

This final frozen-policy gate computes full root Top-B before locking, selects
transparent lock and transition rules on validation only, and reports held-out
QASPER preservation plus the Hotpot entry/locking/propagation decomposition.
The policy implementation is opt-in and does not change one-shot SDK defaults.
Artifacts are under
`../shared/results/paper2_5_iterative_pra/monotonic_adaptive_competition/`.

Contextual query-facet and native-head entry gate:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_query_entry_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_query_entry_facets.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_query_entry_propagation.py --device cuda
```

The first command captures one complete contextual query pass and no independent
window encodings. The second freezes a validation-selected facet/head policy and
reports the matched-budget root matrix. The third runs only because the Hotpot
entry gain cleared the predeclared materiality gate; it reconnects the winning
root to the previously frozen monotonic propagation policy. Artifacts are under
`../shared/results/paper2_5_iterative_pra/query_entry_facets/`.

Facet-type, prompt-support, and grounded-propagation gates:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_grounded_query_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_grounded_facet_gate.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_grounded_propagation_gate.py --device cuda
```

These commands compare parameter-free facet families, capture controlled stale
chat history with an explicit latest-message boundary, and conditionally test
query validation inside native-QK Top-4 successor proposals. The conditional
gate fails on held-out edges, so no end-to-end grounded-closure experiment is
run. Artifacts are under
`../shared/results/paper2_5_iterative_pra/grounded_query_facets/`.
