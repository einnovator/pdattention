# Paper 3.3: Sparse Cross-Document Contextualization

Paper 3.3 begins from the causal boundary established by Paper 3.2: persistent
independent records preserve reusable native K/V, but omit the cross-document
history created by an ordinary packed RAG prefix. This paper asks whether a
small request-specific set of real host-model attention edges can recover that
history.

## Evidence Boundary

- `ESTABLISHED_INHERITED`: Paper 3.2 packed, independent PRA, and rank-8
  residual endpoints, cited with their original provenance.
- `MEASURED_SMOKE`: Paper 3.3 host-observer parity, 0%/100% mask endpoints,
  oracle edge/mass sweeps, and interaction localization on the inception
  cohort.
- `DESIGN_ONLY`: the query-conditioned learned pair selector. Training remains
  locked unless the oracle gate passes.

The canonical split is in
`../shared/results/paper3_3_sparse_crossdoc/splits.json`. It excludes all 30
final Paper 3.2 residual-evaluation identities from Paper 3.3 train,
validation, and test data.

## Inception Decision

The ten-question Qwen3-1.7B-4bit mechanism cohort found:

| Condition | F1 | Official | Physical edges |
| --- | ---: | ---: | ---: |
| Packed RAG | 0.1490 | 0.600 | 100% |
| Independent PRA | 0.1431 | 0.600 | 0% |
| Oracle top-attention | 0.1488 | 0.600 | 0.1% |

The sparse point recovered 97.3% of the small F1 gap, but the absolute
prespecified `0.19/0.67` gate was not met and one ten-question cohort cannot
establish equivalence. Larger budgets were non-monotonic. Learned-selector
training therefore remains locked pending a powered oracle and a task-aware or
ablation-aware importance target.

## Reproduce

See `experiments/paper3_3_sparse_crossdoc/README.md` for split generation,
fixture smoke, and natural oracle commands. Focused tests are:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_sparse_crossdoc.py \
  tests/test_paper3_3_oracle.py \
  tests/test_rag_mlx_native.py \
  tests/test_rag_causal_decomposition.py -q
```

Build the paper from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_3_3.tex
```

Compressed physical teacher graphs are reproducible local artifacts and are
excluded from Git. The tracked manifest preserves source and selection
receipts, graph and plan digests, full condition rows, endpoint parity,
localization summaries, runtime metadata, and plot inputs.
