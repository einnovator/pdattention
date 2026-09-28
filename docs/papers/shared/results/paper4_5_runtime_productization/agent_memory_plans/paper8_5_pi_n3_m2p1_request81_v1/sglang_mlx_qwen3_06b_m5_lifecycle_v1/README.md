# SGLang-MLX lifecycle qualification for Pi N=3 request 81

This directory applies the frozen `frontier_dag_m2_heuristic_p1` selection to
the corrected SGLang-MLX engine path on an Apple M5 16 GB host.  It uses
Qwen3-0.6B-4bit as an engine-mechanics probe; it is not a coding-agent model
quality result.

The receipt clears the declared request-level mechanism contract:

- 37,784 resident source tokens, 13,418 selected original-position K/V tokens,
  and a 2,144-token live wire suffix;
- 35.51% historical K/V retention and 38.98% total realized retention;
- zero selected-history text re-encoding, zero selection-pack bytes, and no
  physical selected-K/V copy;
- exact same-subset logits and two exact continuation token IDs;
- 155,659 bytes of first-layer consumer peak delta, 0.2832% of the
  54,960,128-byte selected-layer K/V extent;
- concurrent borrowing, cancellation, error cleanup, stale-fork rejection,
  offload/restore, termination, and source tombstoning all pass;
- the macOS import guard records no forbidden CUDA/Triton execution.

The 733.99-second elapsed time was collected during a correctness run on a
memory-constrained shared host and is provenance, not a latency claim.  The
larger N=6 fixture fails closed on the same 16 GB host because Metal exhausts
unified memory; that separate admission artifact must not be interpreted as a
correctness failure.
