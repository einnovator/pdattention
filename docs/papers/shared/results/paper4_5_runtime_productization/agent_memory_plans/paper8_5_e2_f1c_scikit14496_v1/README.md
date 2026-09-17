# Paper 8.5 E2+F1C engine handoff

This directory is the immutable Paper 4.5 handoff for the successful
`scikit-learn__scikit-learn-14496` E2+F1C treatment from Paper 8.5 commit
`3aa213b6`.

The nine-request plan was exported without rerunning the selector. The exporter
reconstructed each FULL logical request from the frozen five-episode prefix and
current trajectory, validated the recorded request hashes, then applied the
recorded wire-plan replacements for compact closure receipts. Therefore this
fixture represents the actual model-visible treatment, not an approximation
that omits materialization.

- `frozen_plan.jsonl`: ordered selected resources and original logical record
  positions, segmented into at most 256-token physical resources;
- `request_replay.jsonl`: exact FULL logical requests and frozen generation
  settings for identical-subset engine references;
- `frozen_plan_manifest.json`: source and output hashes.

Logical task success and the 38.53% own-trajectory saving remain Paper 8.5
claims. Paper 4.5 must independently report same-subset exactness, selected
history re-encoding, K/V-copy bytes, consumer temporaries, and lifecycle
correctness for each engine. No request-level engine smoke is an autonomous
task-accuracy result.

## Current engine evidence

Do not substitute the older M2/P1 cross-engine rows for this fixture. Those
rows remain mechanism controls for an earlier policy and do not include the
model-visible compact closure receipts used by E2+F1C.

The receipt-aware HF/CUDA, direct-MLX, and SGLang-MLX request-9 runs are
recorded in `hf_qwen3_06b_request9_mixed_v1.json`,
`mlx_qwen3_06b_request9_mixed_final_v1.json`, and
`sglang_qwen3_06b_request9_mixed_true_offload_v1.json`. All three select the same
17,978 resident original-K/V tokens from a 29,067-token source, explicitly
encode 664 receipt tokens at their original logical positions, and append a
305-token active suffix. After charging the receipts, history saving is 35.87%
and total-visible-token saving is 35.49%; resident-only K/V omission is 38.15%.
All have exact same-subset logits and next tokens, zero selected-K/V copy, and
passing lifecycle gates. MLX and SGLang limit the first-layer disjoint-attention
peak to 172,499 bytes, 0.234% of the 73,637,888-byte selected-layer K/V extent.
The SGLang row additionally verifies that offload
removes the source owner without returning its cache to SGLang's reusable pool,
then restores the exact selected K/V. These are request-level mechanism results,
not autonomous task-accuracy results. vLLM still requires causally positioned
receipt-page construction. Its 16-token page geometry would select 18,016
tokens (38 tokens above the logical plan), predicting 35.73% history and 35.36%
total-visible saving after receipts; those values are not yet measured evidence.
