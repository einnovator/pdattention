# vLLM

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

High-throughput CUDA serving with continuous batching and automatic prefix caching.

## Best PRA deployment today

Selected Context is qualified. Native CUDA correctness and concurrency are promising but matched economics remain pending.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Candidate |
| Native Serving | Candidate |

## Architecture

Controlled native concurrency recovered all expected values without cross-request leakage, but it is not yet a complete selected-versus-native serving comparison.

```text
application -> typed context -> PRA route/select/materialize
            -> vLLM -> generated response
```

## Requirements and tested boundary

- CUDA-capable NVIDIA GPU
- Supported vLLM release
- PRA gateway for typed transport

## Quickstart

```bash
pra runtime doctor -e vllm
pra runtime inspect Qwen/Qwen3-1.7B -e vllm
pra runtime serve Qwen/Qwen3-1.7B -e vllm
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Controlled shared-native concurrency 8 | 48.9 requests/s; 45/45 recoveries; zero leakage | Controlled |
| Matched native economics | NOT_MEASURED | Not measured |

## Metrics and explicit gaps

- **Controlled shared-native concurrency 8:** 48.9 requests/s; 45/45 recoveries; zero leakage  Provenance: `docs/papers/shared/results/paper6_vllm/cuda_connector_concurrency_rtx5060_summary.json`; evidence: Controlled.
- **Matched native economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for production throughput today, especially when prefix caching already captures reuse.

## When Native Memory may help

Reconsider after matched cold, hot, warm, APC, transfer, and tail-latency rows are qualified.

## Limitations

- Native candidate is not the default runtime provider
- Final HBM, transfer, and tail-latency economics are incomplete

## Research evidence

Current public evidence label: **Serving**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Inspect the effective gateway and engine capabilities
- Treat a missing native receipt as Selected Context, not silent native execution

## Production recommendation

Deploy Selected Context; keep Native Memory and Native Serving experimental.
