# Paper 1.5: Positional Semantics for Retrieved Native-KV Memory

This paper studies learned absolute positions and RoPE under bounded encoding, native-KV
retrieval, fragmentation, overlap, relocation, and implicit prompt heads.

Build with latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex in this directory.
Regenerate trained split results with `run_core_experiments.py`. The next-iteration probes are:

```powershell
python experiments/paper1_5_rope/eval_logical_offsets.py --device cuda
python experiments/paper1_5_rope/eval_head_offset_progression.py --device cuda --max-examples 4
python experiments/paper1_5_rope/eval_distance_policy.py --device cuda --iterations 500
python experiments/paper1_5_rope/summarize_next_iteration.py
```

The first three commands record the full code SHA in their JSON/CSV artifacts. The summary
command evaluates the expectations recorded before analysis and does not rerun models.

The research branch is research/paper1-5-rope and the pre-RoPE baseline tag is
paper1-pre-rope-baseline. Results are under docs/papers/shared/results/paper1_5_rope/.
