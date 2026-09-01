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
Engine-native distributed HiCache and fused selected-cache decode remain open.
A caller-owned prefetch hook is implemented, but its five-seed
Python-thread sweep is a negative latency result: all 20 tensors restore
exactly, while requested 10--50 ms leads delay the caller by 193--235 ms.
The replacement event-loop-owned promotion path is now measured: all 35
outputs are exact, and a 250 ms lead reduces median demand stall to 0.013 ms.
The native HTTP gateway is measured through concurrency sixteen, including
streaming cancellation and cleanup. A gateway correction now converts raw
native token rows into OpenAI-compatible deltas and a terminal completion event.
All 31 concurrent outputs are exact; TTFT p95 rises from 35.6 to 5,026.6 ms as
the serialized model runner queues, while throughput stays near 3 requests/s.

The shared lifecycle manager now owns the local built-in HiCache file backend
end to end. Across 80 examples spanning three datasets at 0.6B and QASPER at
1.7B, lossless WARM remains 80/80 exact and recovers after manager restart.
Explicit int8 COLD is exact in only 14/80 sequences; distributed off-node
HiCache and online concurrent cold/warm generation tails remain open.

A controlled two-host WARM bridge now measures off-node transfer separately
from generation. Start the immutable store on one host, forward it over SSH if
macOS local-network privacy blocks the benchmark interpreter, and run the model
host with:

```bash
PYTHONPATH=src:. python experiments/paper6_1_sglang/serve_remote_warm.py \
  --root ~/.cache/pra-remote-warm --port 18161
PYTHONPATH=src:. python experiments/paper6_1_sglang/run_remote_warm_affinity.py \
  --remote-url http://127.0.0.1:18162 \
  --model-host MODEL_HOST --storage-host STORAGE_HOST \
  --transport ssh_tunneled_http \
  --lead-ms 0,10,50,100,250,500,1000,2000,3000 \
  --output docs/papers/shared/results/paper6_1_sglang/offnode_warm_affinity.json
python experiments/paper6_1_sglang/summarize_remote_warm_affinity.py \
  docs/papers/shared/results/paper6_1_sglang/offnode_warm_affinity.json \
  --figure docs/papers/shared/results/paper6_1_sglang/offnode_warm_affinity.png \
  --table docs/papers/shared/results/paper6_1_sglang/generated_offnode_warm_table.tex
```

The measured bridge recovers every tensor exactly. Stable affinity reduces
remote traffic by 3.8--4.0x at concurrency sixteen, but the tunneled path needs
about 2 seconds of prefetch lead to eliminate demand stall. This is controlled
off-node evidence, not a supported SGLang distributed-storage deployment.

A five-example Llama-3.2-1B lifecycle replication is lossless-WARM exact.
Gemma-3-1B is blocked before PRA attachment by the pinned SGLang-MLX backend's
lack of a per-layer sliding-window map.

Run the live lifecycle probe with:

```bash
PYTHONPATH=src:. python -m experiments.paper6_1_sglang.run_live_storage_lifecycle
PYTHONPATH=src:. python -m experiments.engine_serving.summarize_live_storage_lifecycle
PYTHONPATH=src:. python -m experiments.engine_serving.summarize_mac_engine_extension
```
