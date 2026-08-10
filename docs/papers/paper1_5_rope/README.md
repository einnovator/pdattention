# Paper 1.5: Positional Semantics for Retrieved Native-KV Memory

This paper studies learned absolute positions and RoPE under bounded encoding, native-KV
retrieval, fragmentation, overlap, relocation, and implicit prompt heads.

Build with latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex in this directory.
Regenerate model results with `run_core_experiments.py` and deferred-RoPE policy/timing
results with `eval_distance_policy.py --device cuda --iterations 500` under
`experiments/paper1_5_rope/`.

The research branch is research/paper1-5-rope and the pre-RoPE baseline tag is
paper1-pre-rope-baseline. Results are under docs/papers/shared/results/paper1_5_rope/.
