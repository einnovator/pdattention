# Paper 6.2: PRA-MLX

Measured artifacts are under
`docs/papers/shared/results/paper6_2_mlx/`. The serving smoke uses the MLX-LM
HTTP server; the rotating-cache study uses the in-process Python API.

```bash
python -m experiments.engine_serving.summarize
cd docs/papers/paper6_2_mlx
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The paper now includes an in-process native selected-K/V executor, exact
split-cache parity on Qwen, Llama, and Gemma, and 5/5 rotating-local recovery on
Qwen and Llama. Gemma's ordinary and native answer-format controls both score
0/5 despite exact logits, so that row is mechanism parity rather than quality.
Five-seed layer-profile, persistence, and concurrency sweeps are also included,
along with 40-example-per-dataset QASPER and HotpotQA natural-text transport
controls. Those controls test source-dependent native transport, not end-task QA.
