# Paper 6 execution contract

Paper 6 studies whether vLLM should understand PRA logical non-prefix memory,
not merely whether a PRA gateway can call an OpenAI-compatible vLLM server.

Promotion order:

1. Pin vLLM V1, backend, model revision, and cache configuration.
2. Freeze the optimized E0/G10 route-then-materialize baseline.
3. Verify semantic parity with the Hugging Face reference.
4. Preserve Automatic Prefix Caching while PRA resources change.
5. Add first-class logical block identity and lifecycle.
6. Measure residency, prefetch, eviction, sharing, and selected-block execution.
7. Promote a native claim only after actual selected K/V is consumed.

Current status: environment and black-box prefix/PRA smoke measured; logical
block control plane implemented; native selected-K/V execution not measured.

