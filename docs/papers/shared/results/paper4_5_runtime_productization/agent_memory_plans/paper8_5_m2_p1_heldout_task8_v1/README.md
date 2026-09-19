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
retention.  The request preserves original positions and reports:

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
`engine_smokes/hf_qwen25_05b_gtx950m_request1_m2p1_streaming_v1.json`.

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
