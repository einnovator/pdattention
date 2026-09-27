# MLX lifecycle gate for Pi N=6 request 194

This is an engine-mechanics execution of the exact frozen Paper 8.5 plan on a
48 GB M4 Pro with `mlx-community/Qwen3-0.6B-4bit`.  It is not an autonomous
Qwen3-Coder-30B coding-task replay.

Under the engine model's tokenizer, the request contains 77,636 resident source
tokens and a 195-token live suffix.  The unchanged plan selects 23,345
original-position resident K/V entries across 120 logical intervals and five
physical spans.  Total realized retention is 30.245%, or 69.755% omission.

The corrected fused disjoint consumer passes the declared gate:

- zero selected-history token re-encoding and zero selection-pack bytes;
- exact same-subset, dense-engine-oracle, and restored logits (`0.0` maximum
  absolute delta), with next-token ID `151667`;
- 307,235 bytes of first-layer peak delta versus 95,621,120 bytes of selected
  layer K/V (`0.3213%`), with no selected-K/V-sized temporary;
- concurrent borrowing, cancellation, error cleanup, stale-fork rejection,
  eviction/offload, exact restore, termination, and tombstone checks pass.

The 265.71-second end-to-end runner time is correctness/lifecycle provenance,
not a latency-benefit claim.  Model-quality and autonomous-task parity remain
separate gates.

