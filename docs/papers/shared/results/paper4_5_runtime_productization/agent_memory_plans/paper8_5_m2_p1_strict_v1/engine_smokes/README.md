# Frozen-plan request smokes

These rows exercise exact Paper 8.5 M2/P1 request and selection fixtures on
`mlx-community/Qwen3-0.6B-4bit`, MLX-LM 0.31.3, and the 16 GB M5 host. They
are request-level engine-contract probes, not autonomous task solves.

The strict gate requires the candidate to match a dense engine oracle that
uses the identical selected K/V values, original positions, and next wire
token. It also requires zero selected-history re-encoding, no interval-pack
copy, bounded consumer temporary allocation, and all ownership, cancellation,
offload, restoration, and termination checks.

| Artifact | Retention | Wire prefill | Dense-oracle logit delta | Next token | Contract result |
|---|---:|---:|---:|---|---|
| `mlx_qwen3_06b_task4_request1.json` | 45.26% | 16 | 1.000 | exact | quarantined numerical negative |
| `mlx_qwen3_06b_task4_request1_precise.json` | 45.26% | 16 | 1.000 | exact | quarantined numerical negative; precise exponential did not repair it |
| `mlx_qwen3_06b_task4_request2_dtype_match.json` | 45.77% | 16 | 1.125 | exact | quarantined numerical negative; explicit dtype rounding did not repair it |
| `mlx_qwen3_06b_task3_request7_step1.json` | 88.82% | 1 | 0.000 | exact | qualified on Task 3 |
| `mlx_qwen3_06b_task4_request5_step1.json` | 47.92% | 1 | 0.000 | exact | qualified |
| `mlx_qwen3_06b_task4_request5_full_step1.json` | 100.00% | 1 | 0.000 | exact | qualified full-retention control |
| `mlx_qwen3_06b_task5_request9_step1.json` | 52.35% | 1 | 0.000 | exact | qualified on a second task identity |
| `sglang_qwen3_06b_task4_request5_step1.json` | 47.92% | 1 | 0.000 same-consumer | exact | qualified reduced-history lifecycle row |
| `sglang_qwen3_06b_task4_request5_full_step1_v3.json` | 100.00% | 1 | 0.000 same-consumer; 0.9365 fresh-prefill comparator | exact | superseded: comparator changed cache-consumption mode |
| `sglang_qwen3_06b_task4_request5_full_step1_v4.json` | 100.00% | 1 | 0.000 same-consumer and resident-prefix oracle | exact | qualified full-retention control |
| `vllm_metal_qwen3_06b_task4_request5_full_v2.json` | 100.00% | 131-token suffix | same subset and ordinary prefix-cache token exact | exact | qualified full-retention lifecycle row |
| `vllm_metal_qwen3_06b_task4_request5_m2p1_v2.json` | 48.03% total; 47.66% source pages | 131-token suffix | same-subset token exact | exact | qualified reduced-history lifecycle row |

The paired request-5 rows isolate the cause of the earlier strict negatives.
With a 16-token wire-prefill chunk, MLX dispatches the dense reference through
a different prefill kernel than the sparse one-token consumer. At one-token
wire prefill, both paths use the compatible vector-kernel reduction and match
exactly. This is a kernel-dispatch boundary, not evidence that selected K/V
values or original positions are wrong. It does not license a runtime-speed
claim: token-at-a-time prefill is the correctness mode until a matching fused
multi-token consumer is qualified.

For the qualified 47.92% row, 8,617 historical K/V tokens are selected from
18,110 source tokens across 23 disjoint segments. Selected-history
re-encoding and interval-pack bytes are both zero. The measured first-layer
consumer peak delta is 148,061 bytes, 0.42% of the 35,295,232-byte selected
layer K/V extent, so no full-selected-K/V-sized temporary is observed.

The Task 5 row independently selects 11,988/23,077 historical K/V tokens
across 34 segments. Its measured first-layer consumer peak delta is 151,526
bytes, 0.31% of the 49,102,848-byte selected-layer K/V extent. It extends
same-subset correctness to a second task identity, but it is not a second
full-retention control and does not turn these request probes into task-level
agent evidence.

The Task 3 row selects 14,154/15,954 historical K/V tokens across 62
segments at 88.82% total visible retention. Its first-layer peak delta is
153,916 bytes, 0.27% of the 57,974,784-byte selected-layer K/V extent. Thus
the strict reduced-history smoke now covers one request from each of the three
frozen policy tasks, while only Task 4 has the matched full-retention control.

SGLang-MLX consumes the same Task 4 reduced plan with identical geometry. It
is same-consumer logit exact, re-encodes and packs zero selected-history bytes,
and clears the Radix ownership/lifecycle matrix. Its first-layer consumer peak
delta is 157,811 bytes, 0.45% of selected-layer K/V.

The first stronger SGLang full-retention row compared PRA resident-prefix reuse
with an ordinary request that freshly prefetched the whole prompt. It produced
the same two tokens but a 0.9365 maximum logit delta. That row is preserved as
a superseded comparator failure: it changed cache-consumption mode and thereby
reintroduced the multi-token prefill-kernel confound already found in MLX.

The v4 control constructs an independent ordinary resident source cache and
then appends the identical 117-token wire tail. Against that matched no-PRA
prefix-cache baseline, PRA-100 has zero maximum logit delta, emits the same two
tokens, re-encodes and packs zero selected-history bytes, and clears every
Radix lifecycle check. SGLang-MLX is therefore request-level qualified at 100%
for this frozen request. This still does not establish autonomous task-level
parity or runtime benefit.

The vLLM-Metal rows use matched vLLM 0.29.0/vLLM-Metal 0.29.0 packages and
the pinned engine source revision `7390805822b2d7a208b09d55bd07b7572f727e20`
on the 48 GB M4 Pro.  The full control reuses all 18,096 complete source-page
tokens, and the ordinary prefix-cache arm independently reports an 18,096-token
prefix hit; both arms emit token IDs `[151667, 198]`.  The M2/P1 row aliases
8,624 source-page tokens and keeps the same 131-token request suffix, for
48.03% total visible retention and 47.66% page retention.  Page alignment adds
only 21 tokens relative to the exact logical intervals.  Both rows re-encode
zero selected-history tokens, attach existing pages without copying, and add
zero active bytes for two concurrent aliases.  The two lifecycle
materialization events in each artifact belong exclusively to the explicit
lossless offload/restore exercise; request attachment records zero physical
copy events.  Both rows clear the lifecycle matrix.  These remain single-
request correctness and accounting checks, not full frozen-trajectory or
autonomous task qualifications.

The subsequent hash-bound Task 4 sequence closes the request-coverage gap.
All six PRA-100 requests match independent ordinary prefix-cache controls and
all six M2/P1 requests match their identical-subset references.  Across the
reduced sequence, weighted total visible retention is 47.04% (52.96% logical
saving); per-request retention rises from 45.38% to 48.48%.  Complete-page
rounding adds 126 tokens in total, exactly 21 per request.  Both six-request
cells report zero selected-history re-encoding, zero selection-pack bytes, and
zero request-attachment K/V copies.  The immutable request artifacts and
strict reductions are under `../engine_sequences/vllm_metal_task4_full_v3/`
and `../engine_sequences/vllm_metal_task4_m2p1_v3/`.  Their declared claim
boundary remains a frozen request sequence, not an autonomous task solve.
