# Paper 6.9 reproduction

Pinned upstream FreeToken commit:
`3a20a79038338c33bd051c52152e6d1faa4d9791`.

The qualification host was re-audited on 2026-09-08. It has an RTX 5060 Laptop
GPU (8 GiB), driver 592.19, and 25,124,052,992 bytes (23.4 GiB) host memory. A
PyTorch 2.13.0 CUDA 13.0 runtime is operational, but CUDA 13 `nvcc` is still
absent and the documented 20B/30B-class FreeToken configurations remain above
the practical host-memory headroom once the CPU expert pool and runtime are
included. Live model E0/E2/E3 qualification remains blocked, not failed. The
measured re-audit is recorded in `hardware_gate_reaudit_20260908.json`.

The controlled, five-seed scheduling sweep remains runnable without the full
engine stack:

```bash
set PYTHONPATH=src
python experiments/paper6_9_freetoken/run_bandwidth_coordination.py
```

It evaluates a single-link transfer model at 0.5--16 GiB/s. The results test
coordination policy only and are not evidence of a native FreeToken scheduler
integration.

## Dense/MoE reference

`docs/papers/shared/results/paper6_9_freetokens/mac_moe_reference/` vendors the
measured Qwen3-32B and Qwen3-30B-A3B MLX calibration used to separate model
architecture from future FreeToken scheduler effects. The executable runner is
owned by `research/paper4-5-runtime` at commit `0f0ed60`; these results are not
classified as live FreeToken evidence.
