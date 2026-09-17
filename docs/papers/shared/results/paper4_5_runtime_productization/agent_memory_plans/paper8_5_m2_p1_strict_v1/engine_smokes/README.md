# MLX frozen-plan request smokes

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
