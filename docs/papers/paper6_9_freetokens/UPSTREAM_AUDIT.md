# Pinned FreeToken architecture audit

- Revision: `3a20a79038338c33bd051c52152e6d1faa4d9791`.
- Runtime: independent Python/CUDA server with OpenAI and Anthropic APIs.
- Model state: CPU source-of-truth expert pool and shared LRU GPU expert cache.
- Prefill: full-layer double buffering.
- Decode: expert misses are divided between PCIe cache fill and CPU execution by
  the bandwidth-adaptive policy.
- Context state: paged/native KV pools plus radix prefix-cache and recurrent-state
  checkpoints where supported by the model.
- PRA seam: selected semantic K/V must remain a separate identity namespace and
  enter the attention/cache path; expert IDs cannot stand in for PRA resources.
- E3 gate: closed until the real FreeToken scheduler owns PRA prefetch,
  placement, sharing, and eviction.

The audit is based on source, tests, and project documentation at the pinned
revision. It does not claim live behavior for an unexecuted model family.
