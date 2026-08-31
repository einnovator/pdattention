# Paper 6.9 reproduction

Pinned upstream FreeToken commit:
`3a20a79038338c33bd051c52152e6d1faa4d9791`.

The current qualification host has an RTX 5060 Laptop GPU (8 GiB), driver
592.19, and 12 GiB host memory. It lacks CUDA 13 `nvcc`, and the documented
FreeToken model set starts above the practical host-memory budget. Live model
E0/E2/E3 qualification is therefore marked blocked, not failed.

The controlled, five-seed scheduling sweep remains runnable without the full
engine stack:

```bash
set PYTHONPATH=src
python experiments/paper6_9_freetoken/run_bandwidth_coordination.py
```

It evaluates a single-link transfer model at 0.5--16 GiB/s. The results test
coordination policy only and are not evidence of a native FreeToken scheduler
integration.
