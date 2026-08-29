# Paper 6.1: PRA-SGLang

The structured artifacts are in
`docs/papers/shared/results/paper6_1_sglang/`. Regenerate them with:

```bash
python -m experiments.engine_serving.summarize
PYTHONPATH=src:. python -m experiments.paper4_5_runtime.run_storage_lifecycle
cd docs/papers/paper6_1_sglang
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The measured environment uses the pinned SGLang MLX checkout and its documented
loader compatibility patch. The native SGLang-cache path passes five-seed
semantic parity while keeping PRA tokens outside Radix prefix accounting. The
runner hook now passes real prefill and batched decode at concurrency 1--8. It
also includes five-seed QASPER and HotpotQA natural-text transport controls.
Local external HiCache placement, SGLang's built-in file-storage API, and
combined Radix-plus-native execution are measured. A frozen 60-example routed-QA cohort also gives 840/840 exact E0/E2
output pairs across four schedules with 90.6--93.0% fewer visible tokens. Distributed HiCache,
scheduler affinity, online concurrency tails, and fused selected-cache decode
remain open. A caller-owned prefetch hook is implemented, but its five-seed
Python-thread sweep is a negative latency result: all 20 tensors restore
exactly, while requested 10--50 ms leads delay the caller by 193--235 ms.
Engine-native asynchronous transfer is therefore the next prefetch gate.
