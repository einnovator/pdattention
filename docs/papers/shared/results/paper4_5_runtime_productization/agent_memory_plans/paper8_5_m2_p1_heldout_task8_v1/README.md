# Held-out M2/P1 Task 8 native-K/V handoff

This directory is the immutable Paper 4.5 handoff for held-out identity
`django__django-15277`, episode 8 of the completed Paper 8.5 baseline-conditional
campaign frozen at commit `b5e1969f`.  In that declared campaign, the
exact-prefix FULL control and frozen M2/P1 treatment both resolve officially in
six calls.  Along the paired trajectories, message-content input falls from
236,111 to 89,534 tokens (62.08%).  Those are Paper 8.5 logical-policy results,
not native-engine savings.  They are also distinct from Paper 8.5's later
canonical-order unconditional-reliability campaign, whose eighth FULL control
is unresolved; the two campaign orders and trajectory digests must not be
pooled.

The export was produced by Paper 8.5 commit `b5e1969f` from campaign-state
SHA-256 `7c4a47f5c06a966710ec69517fb625054395b692855d221a0ad5049035b22c7a`.
It reconstructs and hash-validates all six original requests; it does not rerun
the selector.  The source prefix contains seven completed issues, making this a
direct test of cross-task resident-K/V retirement rather than a short synthetic
history.

Files:

- `frozen_plan.jsonl`: selected logical records translated to stable resource
  spans;
- `request_replay.jsonl`: exact full logical requests and recorded responses;
- `frozen_plan_manifest.json`: source and output digests;
- `qwen3coder30b_tokenizer_audit.json`: exact Qwen3-Coder tokenizer geometry.

The tokenizer audit realizes 36.46%--38.22% retention across the six requests,
equivalent to 61.78%--63.54% selected-history token saving, without restoring an
excluded record.  Paper 4.5 must now consume these exact fixtures through each
engine's native resident-K/V path and independently report selected-history
re-encoding, physical K/V copy, consumer temporary bytes, position handling,
and lifecycle qualification.  A tokenizer audit alone is not native-K/V
execution evidence.

## HF request-1 execution

The first frozen request now has direct HF/CUDA mechanism evidence on
Qwen2.5-0.5B FP16 and a 4 GB GTX 950M.  The model-specific render contains
37,811 resident source tokens and a 1,951-token request-local suffix.  M2/P1
selects 12,545 source K/V entries, for 36.4569% realized total-history
retention.  The corrected consumer preserves 24 logical record intervals while
coalescing them to seven physical source spans.  The request preserves original
positions and reports:

- zero selected-history re-encoding;
- zero physical K/V copy, interval packing, transient K/V copy, and request-tail
  copy;
- exact same-subset and post-restore logits (maximum delta 0.0);
- 384 bounded streaming-segmented attention calls because Triton is unavailable
  on compute capability 5.0;
- 2,029,289,472 cumulative logical attention-temporary bytes and 464,635,987
  bytes in the separate lossless offload payload;
- every ownership, cancellation, stale-generation, offload/restore,
  termination, and tombstone check passing.

This is one request-level correctness/lifecycle point, not an autonomous task,
latency, or fused-kernel result.  The matched frozen-FULL request is a separate
required control and is not implied by this artifact.  See
`engine_smokes/hf_qwen25_05b_gtx950m_request1_m2p1_coalesced_v2.json`.  The
earlier v1 receipt remains mechanism-correct but predates physical-span
coalescing.

## MLX request-1 execution

The same frozen request now has a direct fused-MLX mechanism point on
`mlx-community/Qwen3-0.6B-4bit` and the 48 GB M4 Pro.  It consumes the same
37,811-token source geometry and 12,545-token M2/P1 selection (36.4569% total
history retention).  The corrected consumer preserves 24 logical record
intervals while coalescing them to seven original-position physical spans.  It
reports:

- zero selected-history re-encoding and zero selection-pack bytes;
- no physical K/V copy and no full-selected-K/V-sized attention allocation;
- 0 active-byte selection delta and a 290,867-byte first-layer attention peak
  delta, 0.5661% of that layer's 51,384,320 selected-K/V bytes;
- exact token and logit agreement with the packed-value, identical-consumer
  oracle (maximum delta 0.0), including after offload/restore;
- all ownership, cancellation, error, stale-generation, offload/restore,
  termination, and tombstone checks passing.

The 4,336,489,740-byte lossless offload payload is reported separately from
selection or attention temporary memory.  This remains a request-level
correctness/lifecycle point; the matched frozen-FULL control, autonomous task,
and latency gates are separate.  See
`engine_smokes/mlx_qwen3_06b_m4pro_request1_m2p1_coalesced_v2.json`.

The first matched FULL diagnostic is retained as a correctness-valid but
performance-invalid bug-discovery receipt.  It passes every mechanism and
lifecycle check at 100% retention, with zero selected-history re-encoding and
copy and exact logits.  However, the consumer treated 192 adjacent logical
record intervals as 192 physical attention segments; the M2/P1 arm similarly
treated 24 logical intervals as 24 segments although their zero-copy geometry
coalesces to seven spans.  FULL took 1,697.71 seconds versus 431.28 seconds for
M2/P1, but that ratio confounds retained tokens with avoidable per-record
launch/reduction overhead and is not a speedup claim.  The engine-neutral plan
and HF/MLX consumers now coalesce adjacent physical views while retaining the
logical record map.  The quarantined diagnostic is
`engine_smokes/mlx_qwen3_06b_m4pro_request1_full_precoalesce_diagnostic_v1.json`.

The post-fix pair closes that request-level control at commit `090a7d52`.
M2/P1 reports 24 logical intervals and seven physical spans; FULL reports 192
logical intervals and one physical span.  Both arms preserve exact tokens and
logits before and after restore, allocate zero bytes during selection, re-encode
zero history tokens, copy/pack zero selected K/V bytes, and pass every lifecycle
check.  Their first-layer attention peak deltas are 290,867 and 323,587 bytes,
respectively, far below their selected-layer K/V extents.  The complete
lifecycle runner takes 500.53 seconds for M2/P1 and 1,111.16 seconds for FULL:
a descriptive 2.22x ratio, or 54.95% elapsed reduction.  This single sequential
pair includes source capture, packed-value oracle work, offload/restore, and all
lifecycle probes; it is not a replicated kernel-latency claim.  The FULL receipt
is `engine_smokes/mlx_qwen3_06b_m4pro_request1_full_coalesced_v2.json`.

## SGLang-MLX request-1 execution

The corrected SGLang-MLX lifecycle bridge now clears the same frozen M2/P1
request on the 48 GB M4 Pro.  It retains 12,545 of 37,811 source-history
tokens (36.4569%), preserves all 24 logical record identities, and coalesces
their resident views to seven original-position physical spans.  The receipt
reports zero selected-history re-encoding, zero selection packing or physical
K/V copy, exact same-subset behavior and exact restoration, and passing
ownership, concurrent-borrower, cancellation, stale-generation,
offload/restore, termination, and tombstone checks.  Selection changes active
memory by zero bytes; the first-layer peak delta is 290,867 bytes, 0.5661% of
the selected-layer K/V extent, with no full-selected-K/V-sized allocation.

The 799.20-second end-to-end lifecycle-runner time includes dense source
capture, reference construction, offload/restore, and lifecycle probes; it is
not a kernel-latency result.  This is a distinct SGLang request/radix-lifecycle
qualification over the same Metal model and hardware as direct MLX, not an
independent model-family replication.  See
`engine_smokes/sglang_mlx_qwen3_06b_m4pro_request1_m2p1_coalesced_v1.json`.

The matched frozen FULL arm at the same `b29b2706` implementation revision
retains all 37,811 source tokens, preserves 192 logical intervals as one
physical span, and also clears exactness, zero-copy/re-encoding, allocation,
and lifecycle gates.  Its first-layer peak delta is 323,587 bytes, 0.2089% of
the 154,873,856-byte selected-layer K/V extent.  The complete lifecycle runner
takes 1,327.73 seconds for FULL versus 799.20 seconds for M2/P1, a descriptive
1.66x ratio or 39.81% reduction.  As with direct MLX, this single sequential
pair includes source capture, reference work, offload/restore, and lifecycle
probes and is not replicated kernel latency.  The FULL receipt is
`engine_smokes/sglang_mlx_qwen3_06b_m4pro_request1_full_coalesced_v1.json`.

## vLLM/CUDA request-1 execution

The same receipt-free M2/P1 request now clears the bounded vLLM 0.28 native
page-alias path on a GeForce RTX 5060 Laptop GPU with Qwen3-0.6B FP16.  The
16-token cache granularity rounds 12,545 logical selected-history tokens to
12,656 resident tokens, an overhead of 111 tokens.  Including the 1,938-token
wire suffix, the request exposes 14,594 rather than 39,762 tokens, for 63.30%
total-visible-token saving and 66.56% resident-original-K/V omission.  Two
repeated continuations emit token IDs `[151667, 198]` and the engine reports:

- two authoritative scheduler-page aliases installed and released;
- zero selected-history re-encoding, physical K/V copy, and host-to-device
  traffic;
- zero receipt encoding, host packing, receipt transfer, or copy-on-write,
  because this policy contains no compact closure receipts;
- complete registry cleanup after both borrowers.

The runner previously rejected receipt-free policies before page attachment;
commit `d4e1568a` fixes that harness defect by aliasing selected canonical pages
directly rather than manufacturing a materialized-history source.  This is a
request-level mechanism point, not an autonomous-task or runtime-economics
result.  See
`engine_smokes/vllm_cuda_qwen3_06b_rtx5060_request1_m2p1_pagealias_v1.json`.

The matched FULL control was added and both arms were repeated at commit
`2f168e0e`.  FULL aliases all 37,824 resident page tokens (37,811 logical
source tokens plus the 13-token terminal-page fill), exposes the same 39,762
visible tokens as the unabridged request, emits the same `[151667, 198]`
continuation twice, and retains zero copy/re-encoding and complete cleanup.
Mean post-alias two-token generation time is 0.296 seconds for M2/P1 versus
0.694 seconds for FULL, a descriptive 2.35x ratio.  These are two borrowers
inside one run per arm after engine initialization and source capture, not a
replicated end-to-end latency estimate.  The FULL receipt is
`engine_smokes/vllm_cuda_qwen3_06b_rtx5060_request1_full_pagealias_v1.json`.
