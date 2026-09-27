# Frozen Pi M2/P1 request 194

This bundle transfers the final request of the completed Paper 8.5 Pi 0.75.3
persistent `N=6` campaign into Paper 4.5 without rerunning selection.  The
source policy is `frontier_dag_m2_heuristic_p1`; the model-visible request and
selected-message identities are bound by the export and reconstruction
manifests.

The source request contains 388 messages.  Paper 8.5 counts 59,331 logical
message-content tokens, materializes 16,756, and therefore omits 71.76% on
this request.  The exported strict subset selects 122 messages, has three
mandatory current-state messages, and contains no compacted or newly encoded
replacement text.

The exact Qwen3-Coder tokenizer audit maps the same frozen decision to:

- 77,955 full prompt tokens;
- 77,756 resident source-history tokens plus a 199-token wire tail;
- 21,912 selected original-position resident K/V tokens;
- 28.36% total realized retention, or 71.64% token/K/V opportunity;
- 120 logical intervals coalesced into five physical spans.

The full-retention control maps the same request to one 77,756-token resident
source interval and the identical 199-token wire tail.  These artifacts prove
request identity, strict-subset semantics, and tokenizer geometry only.  They
do not prove engine correctness, zero-copy consumption, lifecycle safety,
latency benefit, or autonomous task parity.  Each native engine must consume
the frozen plan unchanged and report those gates independently.

The first vLLM/CUDA admission on an RTX 5060 8 GB fails closed before
inference.  The prompt exceeds Qwen3-0.6B's declared position limit and the
full FP16 source K/V alone exceeds physical VRAM.  The raw receipt and the
unexecuted page-rounded geometry are under
`vllm_cuda_rtx5060_admission_v1/`; they are a hardware/model admission result,
not a native N=6 execution.

The MLX mechanics probe under `mlx_qwen3_06b_m4pro_lifecycle_v1/` does execute
the exact frozen plan on Qwen3-0.6B-4bit.  Under that engine tokenizer it omits
69.755% of total visible tokens, re-encodes and packs zero selected history,
is same-subset/dense-oracle/restored-logit exact, and passes the full declared
lifecycle suite.  It remains a small-model mechanism result, not autonomous
Qwen3-Coder-30B task parity or a latency-benefit result.
