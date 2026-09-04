# Paper 3.2 experiment ladder

The first gate is a dependency-light mechanism smoke. It validates frozen
candidate/selection receipts and the RAG+PRA position/profile matrix over five
synthetic corpus seeds. It does **not** run a language model and therefore does
not report answer quality.

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
python -m experiments.paper3_2_rag.run_mechanism_smoke
```

Setting `PYTHONPATH` is required when multiple PRA worktrees have editable
installs; it prevents Python from resolving a different paper branch.

Outputs are written to
`docs/papers/shared/results/paper3_2_rag/mechanism_smoke/`.

Subsequent gates add Qwen3-0.6B answer generation, natural datasets, external
retrieval services, and only then larger models. Candidate retrieval and
realization always meet at immutable receipts.
