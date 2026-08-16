# Paper 4 compute budget

## Benchmark-before-budget decision

Device: NVIDIA GeForce GTX 950M, 4 GB VRAM, CUDA PyTorch 2.12.1+cu126.

A one-seed calibration used the final 64D/8-layer architecture with 20 base
steps and 10 adaptation steps. Baseline throughput was approximately
8.5k--9.9k physical tokens/s. Differentiable native-KV adaptation ranged from
3.3k to 4.5k physical tokens/s across LoRA and full-weight regimes.

The five-seed controlled schedule was therefore approved with:

- 800 GlobalSA and 800 LocalSA steps per seed;
- 500 steps for each converted trainable rung;
- 1,300 steps for native PRA from scratch;
- batch size 32, 4,096 deterministic training examples, and 512 held-out tests;
- private checkpointing after every model/seed boundary.

Projected wall clock on the measured device was about two hours. The completed
five-seed schedule used 8,303 seconds (2.31 hours) inside training calls, plus
checkpoint evaluation and artifact generation. Mean measured throughput was
9.33k physical tokens/s for both SA baselines and 3.80--3.99k physical tokens/s
for trainable PRA regimes. Sampled device residency during the run was about
0.52 GB; this is an observation from `nvidia-smi`, not an instrumented peak-memory
measurement. Private checkpoints ranged from 1.75 to 2.19 MiB.

This is a
causal diagnostic pilot, not the 10M-token scaling point. Gate 0 subsequently
passed three criteria. Tier 1 is scientifically unblocked but requires its own
100--500-step device benchmark and budget decision; Gemma remains a later gate.

## Escalation rule

Proceed only if at least two preregistered Gate 0 signals improve. Otherwise,
retain the negative result and revise the architecture before spending a
larger-model budget.
