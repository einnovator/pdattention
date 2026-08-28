# Paper 4.5 engine-feature applicability

| Feature | Paper 4.5 action | Evidence or boundary |
|---|---|---|
| HF eager semantic reference | Implement and test | Visible-prefix/native logits, E0-E3 geometry, and decode-lifetime gates |
| Modern-GPU `torch.compile` | Measure when available | Current GTX 950M is compute capability 5.0; no result is inferred |
| Triton/custom CUDA | Conditional | Requires profiler evidence after an eager and compile baseline |
| Async transfer/prefetch | Implemented configuration, measurement deferred | Pinned and non-blocking transfer controls exist; no modern GPU is available |
| Gateway streaming | Implement and test | Portable HF iterator and OpenAI-compatible SSE with cancellation cleanup |
| E3 scheduler | Defer | Scheduler semantics belong to dedicated engine integrations |
| Continuous batching | Defer | Standard HF generation is not a production continuous-batching service |
| p95/p99 serving | Defer | No representative long-running production service is installed |
| Cache/page scheduling | Generic cache only | Tenant-scoped byte-bounded LRU is tested; page ownership is engine-specific |
| Multi-tenant physical K/V | Logical/runtime isolation | Tenant/user/session cache keys and per-tenant eviction are tested |
| High-throughput multi-session | Correctness only | HF requests are serialized around temporary adapter state |

HF is the complete semantic reference. The gateway and wire contract are reusable;
SGLang, vLLM, FreeToken, llama.cpp, and MLX native integrations are separate workstreams.
