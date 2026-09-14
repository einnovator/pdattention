# AGENTS.md — Paper 6.7: PRA on llama.cpp

## Portable native semantic K/V for the GGUF/local-inference ecosystem

### Central question
Can PRA become native in the GGUF/local-inference ecosystem using public cache/state APIs where possible, or a minimal ggml/llama memory extension where necessary?

Reuse Paper 4.5 contracts. PRA owns records, routing/selection, profiles, task/session semantics, authorization and HOT/WARM/COLD/SOURCE. llama.cpp owns GGUF execution, device placement, KV/state representation, graph execution, server scheduling and hardware kernels.

### Why it matters
Target local/private AI, CPU inference, consumer CUDA, Apple Metal, Vulkan/heterogeneous hardware, GGUF quantized models and edge deployments.

### Work plan
1. Pin llama.cpp commit/version and audit GGUF, sequence/slot state, KV/cache save/restore APIs, server scheduling and public extension seams.
2. Run common FULL vs E0_SELECTED qualification through llama-server/OpenAI-compatible serving.
3. Determine whether public state/KV APIs can represent truly detached immutable selected-resource K/V. Do not call ordinary sequential state restore E2.
4. If needed, implement the smallest llama/ggml extension accepting detached selected per-layer K/V with one joint attention normalization.
5. Run geometry-matched correctness: no-PRA, full context, cached source, E2 selected resource, multiple resources, persistent decode, wrong-memory and absent-memory controls.
6. Integrate PRAStorageManager: HOT attention-ready K/V, lossless WARM, experimental COLD, SOURCE reconstruction.
7. Test multi-slot/session isolation, identical visible queries with different PRA resources, cleanup and safe immutable sharing.
8. Qualify CPU, CUDA, Metal and Vulkan independently where available. Never infer one backend from another.
9. Test small Qwen/Llama/Gemma GGUFs then at least one larger quantized model.
10. Compare FULL, E0_SELECTED, E2_HOT, E2_WARM and SOURCE on quality, TTFT/ITL, throughput, memory, K/V bytes and successful req/s.
11. Claim E3 only if llama-server itself owns PRA scheduling/prefetch/placement/sharing/eviction.

### Required metrics
Task quality/F1/EM, exact parity, gold logP where feasible, source/visible/selected tokens, active K/V bytes, TTFT/ITL/completion p50/p95/p99, req/s, output tok/s, CPU/GPU memory, cache occupancy/hits, restore/prefill cost, evictions/reloads, shared bytes saved.

### Paper outputs
Tables: backend capability; E0/E2 correctness; GGUF/model compatibility; memory-latency frontier; shared reuse; backend qualification.
Figures: architecture; quality-memory frontier; TTFT pressure; memory vs context; E0/E2 ratio; portability matrix.

### Editorial structure
Introduction; PRA primer; llama.cpp/GGUF architecture; cache/state model; PRA integration; correctness; lifecycle/isolation; cross-backend experiments; related work; limitations; reproducibility.

### Stop gate
E0 complete; public APIs honestly classified; E2 implemented or precisely blocked; matched correctness and isolation pass; CPU plus one accelerator measured; physical metrics captured; product-matrix rows emitted; tests/PDF pass.

### Core message
PRA should add portable semantic non-prefix resource identity without replacing llama.cpp's portable inference machinery, using the shallowest integration level that measurably improves quality-adjusted local inference economics.
