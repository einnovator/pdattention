# Paper 6.1: PRA-SGLang

The structured artifacts are in
`docs/papers/shared/results/paper6_1_sglang/`. Regenerate them with:

```bash
python -m experiments.engine_serving.summarize
cd docs/papers/paper6_1_sglang
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The measured environment uses the pinned SGLang MLX checkout and its documented
loader compatibility patch. The native SGLang-cache path passes five-seed
semantic parity while keeping PRA tokens outside Radix prefix accounting. The
runner hook now passes real prefill and batched decode at concurrency 1--8. It
also includes five-seed QASPER and HotpotQA natural-text transport controls.
Combined Radix-plus-native scheduling and HiCache placement remain open gates.
