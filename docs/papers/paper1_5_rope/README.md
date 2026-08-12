# Paper 1.5: Positional Semantics for Retrieved Native-KV Memory

This paper studies learned absolute, sinusoidal, and RoPE positions under bounded encoding,
native-KV retrieval, fragmentation, overlap, relocation, and implicit prompt heads.

Build with latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex in this directory.
Regenerate trained split results with `run_core_experiments.py`. The next-iteration probes are:

```powershell
python experiments/paper1_5_rope/eval_logical_offsets.py --device cuda
python experiments/paper1_5_rope/eval_head_offset_progression.py --device cuda --max-examples 4
python experiments/paper1_5_rope/eval_distance_policy.py --device cuda --iterations 500
python experiments/paper1_5_rope/run_pooling_geometry.py --device cuda
python experiments/paper1_5_rope/summarize_next_iteration.py
```

The first three commands record the full code SHA in their JSON/CSV artifacts. The summary
command evaluates the expectations recorded before analysis and does not rerun models.

The pooling command reuses frozen RoPE checkpoints and runs the five-seed tiny/small
HotpotQA/QASPER matrix. It compares post-/pre-RoPE means, centered subgists at
`G=1,2,4,8`, raw hidden cosine, and the established learned 32-D router. Results and plots are
under `docs/papers/shared/results/paper1_5_rope/pooling_geometry/`.

The research branch is research/paper1-5-rope and the pre-RoPE baseline tag is
paper1-pre-rope-baseline. Results are under docs/papers/shared/results/paper1_5_rope/.

## Final Validation

From the repository root, reproduce one tiny sinusoidal path first:

```powershell
python experiments/paper1_5_rope/train_validation_checkpoints.py --device cuda `
  --tiers tiny --position-modes sinusoidal --seeds 1 --smoke
python experiments/paper1_5_rope/run_wikitext_validation.py --device cuda `
  --tiers tiny --position-modes sinusoidal --seeds 1 --smoke
python experiments/paper1_5_rope/run_qa_validation.py --dataset hotpotqa --device cuda `
  --tiers tiny --position-modes sinusoidal --seeds 1 --smoke
```

The full five-seed, two-tier run is resumable:

```powershell
python experiments/paper1_5_rope/train_validation_checkpoints.py --device cuda
python experiments/paper1_5_rope/eval_logical_offsets.py --device cuda `
  --position-modes absolute sinusoidal rope `
  --output-dir docs/papers/shared/results/paper1_5_rope/validation `
  --result-name positional_mechanism_offset_validation
python experiments/paper1_5_rope/eval_head_offset_progression.py --device cuda `
  --position-modes absolute sinusoidal rope `
  --output-dir docs/papers/shared/results/paper1_5_rope/validation `
  --result-name capacity_validation
python experiments/paper1_5_rope/run_wikitext_validation.py --device cuda
python experiments/paper1_5_rope/run_qa_validation.py --dataset hotpotqa --device cuda
python experiments/paper1_5_rope/run_qa_validation.py --dataset qasper --device cuda
python experiments/paper1_5_rope/summarize_night_validation.py
```

On the recorded NVIDIA GTX 950M, WikiText completed in about three minutes, HotpotQA in about
ten minutes, and QASPER in about nine minutes after checkpoint reuse where available. First-time
training and different GPUs will change these times. Checkpoints are under
`out/paper1_5_rope/`; full JSON, CSV, plots, expected-versus-observed records, and the recursive
manifest are under `docs/papers/shared/results/paper1_5_rope/validation/`.

Build the paper in this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```
