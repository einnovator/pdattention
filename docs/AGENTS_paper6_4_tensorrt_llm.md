# AGENTS.md --- Paper 6.4: PRA on NVIDIA TensorRT-LLM

Reuse Paper 4.5 semantic/runtime contracts. TensorRT-LLM already
provides paged KV cache, prefix reuse, priority eviction, in-flight
batching, KV Cache Connector, OpenAI-compatible `trtllm-serve`, and
disaggregated prefill/decode. Do not build a second PRA runtime inside
it.

## Research question

Can PRA semantic non-prefix memory become a first-class TensorRT-LLM KV
resource while preserving paged-context attention, prefix reuse,
in-flight batching, KV connectors and disaggregated serving?

## Plan

1.  **Environment.** Record GPU/CC/VRAM, CUDA/driver, TensorRT/TRT-LLM,
    model/precision/KV precision, TP/PP/EP, paged-context FMHA, block
    size and memory fraction.
2.  **E0 baseline.** Use `trtllm-serve` and common frozen manifests:
    no-PRA, prefix reuse, selected text, prefix+selected text, full
    context. Capture `/metrics`.
3.  **KV architecture audit.** Map `KVCacheManager`, paged blocks,
    prefix search/reuse, priority eviction, cache groups, request
    lifecycle and in-flight batching. Keep
    `prefix identity != PRA identity != physical block identity`.
4.  **KV Cache Connector.** Prefer the official connector for
    WARM/external storage. Integrate strict PRA fingerprints,
    authorization, sharing, HOT/WARM/COLD/SOURCE.
5.  **Native E2 attachment.** Attach selected PRA blocks without
    pretending they are ordinary sequential prefix blocks. Preserve
    positions, masks, cache-group topology, request pinning, cleanup,
    one-copy attachment and isolation.
6.  **Geometry-matched parity.** E0/E2 must derive source K/V from
    matched TensorRT-LLM execution geometry. Measure first-token/logit
    agreement where accessible, exact sequence, F1, gold log-prob,
    multiple resources/long intervals/cache groups.
7.  **In-flight batching.** Sustained arrivals; measure
    TTFT/TPOT/completion p50/p95/p99, req/s, output tok/s, queue time,
    batch occupancy, preemption/cancellation, block occupancy,
    hits/evictions/reloads.
8.  **HBM/connector economics.** Measure model/KV/PRA HBM, WARM bytes,
    H2D/D2H, PCIe/NVLink, shared bytes avoided, promotion, overlap and
    reload amplification.
9.  **Priority eviction.** Compare native/default eviction with
    PRA-derived physical priority hints. Semantic policy remains in PRA.
10. **Disaggregated serving.** Use current supported context/decode
    disaggregation (prefer NIXL when available). Test SOURCE/WARM/HOT
    PRA resource paths, transfer/layout conversion, overlap, global
    request/resource identity and authorization.
11. **TP/multi-GPU.** Test TP=2 where possible; include topology in
    fingerprints.
12. **Quantized KV.** Compare native FP/BF16, TensorRT-LLM INT8/FP8 HOT
    KV where supported, lossless PRA WARM/COLD, and experimental
    persistent quantization. Measure semantics; do not assume parity.

## Required metrics

Quality/parity; selected/visible tokens; TTFT/TPOT/completion tails;
req/s/tok/s; queue/batch occupancy; HBM/host/WARM bytes; connector hit
rate; H2D/D2H/NVLink/PCIe; promotion/overlap; evictions/reloads;
preemptions; shared bytes saved.

## Tables

Capability mapping; E0/E2 parity; prefix reuse × PRA state; online
serving; HBM/connector economics; disaggregated serving; KV precision.

## Figures

Scheduler/cache architecture; TTFT p99 vs load; HBM vs concurrency;
connector transfer/promotion; shared-resource savings; disaggregated
transfer overlap.

## Editorial structure

Introduction; minimal PRA; TensorRT-LLM KV/scheduler architecture;
semantic-resource mapping; connector/native integration; correctness;
online serving; disaggregated serving; quantized KV; related work;
limitations; reproducibility.

## Stop gate

E0 reproducible; official connector integrated where supported; E2
implemented or precisely blocked; geometry parity measured; in-flight
batching tails and HBM/transfer telemetry captured; shared concurrency
measured; disaggregated path tested if hardware permits; product matrix
emitted; tests/PDF pass.

## Core message

TensorRT-LLM already has a sophisticated physical KV system. PRA should
contribute semantic, query-addressed non-prefix identity and lifecycle
policy while reusing TensorRT-LLM paging, batching, connectors, eviction
and KV transfer.
