# AGENTS.md — Paper 3

## Goal
Write the runtime systems paper for PRA.

## Main Contribution
Describe how PRA could become native to inference engines by mapping reference handles to paged KV memory blocks and dynamically expanding attention metadata.

## Must Cover
- KV cache basics.
- vLLM/PagedAttention concepts.
- Layer-specific cache keyed by ref_id and layer_id.
- Block-table expansion.
- Scheduler implications.
- Cache eviction.
- Latency and throughput tradeoffs.
- RoPE/position handling for separately encoded memory.
- Cross-attention memory vs direct KV injection.

## Important
This paper is systems-oriented and should be careful about what is implemented versus future engineering.
