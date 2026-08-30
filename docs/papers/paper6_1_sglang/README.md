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
output pairs across four schedules with 90.6--93.0% fewer visible tokens.
An expanded 149-unique-question confirmation gives 2,086/2,086 exact pairs;
its cold/warm/multi-query/concurrent E2/E0 ratios are
0.980/1.074/1.071/1.054.
Distributed HiCache, scheduler affinity, and fused selected-cache decode remain
open. A caller-owned prefetch hook is implemented, but its five-seed
Python-thread sweep is a negative latency result: all 20 tensors restore
exactly, while requested 10--50 ms leads delay the caller by 193--235 ms.
The replacement event-loop-owned promotion path is now measured: all 35
outputs are exact, and a 250 ms lead reduces median demand stall to 0.013 ms.
The native HTTP gateway is also measured through concurrency eight, including
streaming cancellation and cleanup; TTFT p95 rises from 40.4 to 2,336.9 ms as
the serialized model runner queues.

The shared lifecycle manager now owns the local built-in HiCache file backend
end to end. Across 80 examples spanning three datasets at 0.6B and QASPER at
1.7B, lossless WARM remains 80/80 exact and recovers after manager restart.
Explicit int8 COLD is exact in only 14/80 sequences; distributed off-node
HiCache and online concurrent cold/warm tails remain open.

A five-example Llama-3.2-1B lifecycle replication is lossless-WARM exact.
Gemma-3-1B is blocked before PRA attachment by the pinned SGLang-MLX backend's
lack of a per-layer sliding-window map.

Run the live lifecycle probe with:

```bash
PYTHONPATH=src:. python -m experiments.paper6_1_sglang.run_live_storage_lifecycle
PYTHONPATH=src:. python -m experiments.engine_serving.summarize_live_storage_lifecycle
PYTHONPATH=src:. python -m experiments.engine_serving.summarize_mac_engine_extension
```
