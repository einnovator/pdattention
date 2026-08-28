# Paper 3: Toy-First Causal K/V Materialization

Build from this directory with:

```powershell
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Run the controlled five-window/five-seed study before pretrained confirmation:

```powershell
python -m experiments.paper3_kv_materialization.run_toy_materialization --device cuda
python -m experiments.paper3_kv_materialization.summarize_toy_materialization
```

The controlled runner reuses frozen Paper-2.5 checkpoints. It fixes conceptual
selection to one oracle parent and varies only the physical native-K/V detail.
Its checkpoint JSONL is resumable but intentionally not part of the paper
artifact set.

The larger Qwen3-0.6B confirmation uses a validation-frozen compact policy set:

```powershell
python -m experiments.paper3_kv_materialization.run_oracle_frontier `
  --phase validation --study confirmation `
  --artifact-prefix pretrained_confirmation --examples-per-dataset 16 `
  --max-new-tokens 8 `
  --output-dir docs/papers/shared/results/paper3_kv_materialization/pretrained_confirmation `
  --policy-selection docs/papers/shared/results/paper3_kv_materialization/pretrained_confirmation/pretrained_policy_selection.json `
  --musique-dev D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl `
  --twowiki-dev D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets/2wiki/dev.json
python -m experiments.paper3_kv_materialization.run_oracle_frontier `
  --phase heldout --study confirmation `
  --artifact-prefix pretrained_confirmation --examples-per-dataset 32 `
  --max-new-tokens 8 `
  --output-dir docs/papers/shared/results/paper3_kv_materialization/pretrained_confirmation `
  --policy-selection docs/papers/shared/results/paper3_kv_materialization/pretrained_confirmation/pretrained_policy_selection.json `
  --musique-dev D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl `
  --twowiki-dev D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets/2wiki/dev.json
python -m experiments.paper3_kv_materialization.summarize_pretrained_confirmation
```

The original oracle pilot and Paper-2.5 selector factorial remain preserved.
The tracked `paper.pdf` is built from both the controlled causal study and the
pretrained confirmation.

Run the corrected layer-placement reconciliation after the pretrained frontier:

```powershell
python -m experiments.paper3_kv_materialization.run_layer_reconciliation `
  --examples-per-dataset 16 `
  --output-dir docs/papers/shared/results/paper3_kv_materialization/layer_reconciliation `
  --musique-dev D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl `
  --twowiki-dev D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets/2wiki/dev.json
python -m experiments.paper3_kv_materialization.summarize_layer_reconciliation `
  --root docs/papers/shared/results/paper3_kv_materialization/layer_reconciliation
```

The reconciliation uses corrected source-offset query positions and keeps
native source K/V at their source-relative RoPE positions. It separates
address, detail-K/V storage, routing, and consumer roles; the sweep varies only
the consumer set while holding oracle identity and transport fixed.
