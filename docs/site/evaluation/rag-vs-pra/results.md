# Results

Paper 4.5 uses MultiHop-RAG as a controlled engine workload. Retrieval,
selector, transfer, and answer-quality conclusions belong to Papers 3.2 and
3.3 and are not repeated here. All comparisons below freeze the selected
document intervals before comparing their engine realization.

## MLX engine modes

On the 50-question, 2K cohort, Selected Context and Native Memory produce
identical outputs. Selected Context serializes the frozen selection as visible
text and re-prefills it per request. Native Memory encodes the same selection as
detached MLX K/V. Cold measurements include native encoding; warm Native Memory
reuses retained immutable K/V.

| Qwen3-4B 4-bit mode | Regime | Mean TTFT | TTFT p95 | Total latency | Visible tokens | Native tokens | Reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected Context | Cold | 2,633 ms | 2,716 ms | 3,024 ms | 2,105 | 0 | 0.00 |
| Native Memory | Cold | 2,648 ms | 2,701 ms | 3,121 ms | **79** | 2,026 | 0.00 |
| Selected Context | Warm | 2,640 ms | 2,728 ms | 3,031 ms | 2,105 | 0 | 0.00 |
| Native Memory | Warm | **166 ms** | **211 ms** | **638 ms** | **79** | 2,026 | **1.00** |

Cold Native Memory is 3.2% slower end to end. After retaining the same
selection, it reduces total latency by 78.9%, raises throughput from 10.56 to
50.21 tokens/s, and uses about 299 MB of active K/V. Peak unified memory is
0.63 GB below the repeatedly prefilled Selected Context path.

## Budget and model scaling

| MLX-LM model | Budget | Cold Native vs Selected | Warm Selected to Native | Mean TTFT, warm | Active K/V | Output parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-4B 4-bit | 2K | +3.2% | 3.031 to 0.638 s (-78.9%) | 2.640 to 0.166 s | 299 MB | Exact |
| Qwen3-4B 4-bit | 4K | +3.3% | 6.121 to 0.799 s (-86.9%) | Not reported | 602 MB | Exact |
| Qwen3-8B 4-bit | 2K | +1.8% | 5.541 to 0.990 s (-82.1%) | 4.908 to 0.276 s | 285 MB | Exact |

These are repeated-selection results, not comparisons with an ordinary
selected-text prefix-cache hit. They qualify contiguous realization only for
the pinned MLX-LM models and host.

## Persistent chunk mode

Changing the reuse unit to independently encoded 256-token chunks produces a
different boundary. Across 50 changing questions, MLX reuses 22.0% of chunk
lookups and reduces cumulative request time from 151.19 to 131.78 seconds
(-12.8%). Mean TTFT falls from 2.633 to 2.163 seconds, but retained detail grows
to 6.38 GB and outputs are not exact because chunks retain source-local
post-RoPE positions. This mode remains unqualified pending position-aware
composition.

## Precision-specific one-shot cost

On one M5 host with Qwen3-4B, one-shot Native Memory costs 5.8--6.1% more total
latency than the deployed No-PRA pipeline for BF16, 10.0--10.7% more for INT8,
and 14.8--16.1% more for INT4. This historical ten-question cohort changes both
selector and representation, so its quality columns are guardrails rather than
a selector-frozen transport result.

## Claim boundary

Paper 4.5 supports three engine conclusions:

- frozen contiguous MLX Native Memory is output-exact in the tested 4B and 8B modes;
- retained K/V can remove repeated selected-text prefill, while cold native encoding is slightly slower;
- independently cached chunks trade lower runtime for changed semantics and are not qualified.

The detailed candidate curves, retriever comparisons, learned and lexical
selectors, cross-model transfer, and answer-quality analysis are intentionally
left to Papers 3.2 and 3.3.
