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

## Current autonomous evidence

The only fresh post-fix autonomous row is
`autonomous_llamacpp_qwen3coder30b_postfix_v1`. Two FULL controls and PRA-100
all solve in eight calls with exact request, response, and action hashes.
Native E2+F1C also solves in eight calls, saves 39.88% of cumulative
materialized message-content tokens, and omits 40.23% of weighted steady-state
resident K/V. Selected-history re-encoding and K/V-copy bytes are both zero.
This is a single-task, single-engine result. Fresh autonomous HF, MLX, SGLang,
and vLLM runs are pending.

## Quarantined pre-final request-level evidence

Do not substitute the older M2/P1 cross-engine rows for this fixture. Those
rows remain mechanism controls for an earlier policy and do not include the
model-visible compact closure receipts used by E2+F1C.

The engine files below were captured before the final autonomous source-
bootstrap contract and later engine corrections. They remain reproducibility
artifacts but are not current cross-engine savings or agent-accuracy evidence;
each engine must rerun the frozen FULL, PRA-100, and E2+F1C sequence.

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
not autonomous task-accuracy results.

The receipt-aware vLLM/CUDA request-9 run is recorded in
`vllm_cuda_request9_mixed_final_v1.json`. It causally encodes all six old-
position receipts, page-rounds 17,978 logical selected-history tokens to 18,016
resident tokens, and moves five active-tail tokens into the otherwise partial
terminal source page. It therefore measures 35.75% historical saving after
receipts and 35.38% total-visible saving, 0.11 percentage point below the
interval-addressed engines. Two final requests both emit token IDs
`[151667, 198]`, matching HF and SGLang; the first token also matches MLX,
whose frozen run generated one continuation token. Original selected history
is neither re-encoded nor copied, and all scheduler sources and aliases are
released. Receipt construction is a correctness path rather than a runtime
claim: it performs 76.15 MB D2H capture, 239.93 MB host packing, 240.27 MB H2D
reload, and 12.85 MB receipt-page copy-on-write. The adjacent
`vllm_cuda_receipt_capture_smoke_v1.json` is the smaller primitive smoke.

The receipt-aware llama.cpp/Metal request-9 run is recorded in
`llamacpp_qwen3_14b_request9_mixed_v1.json`.  It uses the same frozen Qwen3
token geometry with a Qwen3-14B-Q4_K_M consumer, retains the same 17,978
original-position K/V tokens, encodes the same 664 receipt tokens, and appends
the same 305-token suffix.  It therefore realizes 35.87% history saving after
receipts, 35.49% total-visible saving, and 38.15% resident-original-K/V
omission.  Two executions after request-slot cleanup both emit
`[151667, 198]`; selected history is neither re-encoded nor physically copied,
and source/request cleanup passes.  Qualification required two additional
engine corrections: causal interleaving of resident-range attachment with
old-position receipt evaluation, and forward sparse-position advancement over
omitted gaps.  The latter continues to reject overlap, reversal, and
non-contiguous positions within one input batch.  The heavily interrupted host
run is mechanism evidence only; its timing fields are not used for a runtime
claim.
