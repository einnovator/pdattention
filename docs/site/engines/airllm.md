# AirLLM

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Layer-streamed inference for models that exceed accelerator memory.

## Best PRA deployment today

Selected Context; it reduces visible work without adding a separate native reference pass.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Research only |
| Native Serving | Not measured |

## Architecture

Native memory is semantically feasible but slower in the measured request path, including warm reuse.

```text
application -> typed context -> PRA route/select/materialize
            -> AirLLM -> generated response
```

## Requirements and tested boundary

- CUDA system supported by AirLLM
- Sufficient host or storage capacity for streamed layers
- PRA gateway or harness

## Quickstart

```bash
pra gateway serve --mode selected-context --backend custom --backend-url http://127.0.0.1:8000/v1
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Native/selected TTFT | 2.186x-2.351x | Natural workload |
| Native/selected ITL or completion | approximately 1.10x-1.17x | Natural workload |
| Mean separate reference encoding | 10.91 seconds | Natural workload |

## Metrics and explicit gaps

- **Native/selected TTFT:** 2.186x-2.351x  Provenance: `docs/papers/shared/results/paper6_6_airllm/tinyllama_cuda_natural_summary.json`; evidence: Natural workload.
- **Native/selected ITL or completion:** approximately 1.10x-1.17x  Provenance: `docs/papers/shared/results/paper6_6_airllm/tinyllama_cuda_natural_summary.json`; evidence: Natural workload.
- **Mean separate reference encoding:** 10.91 seconds  Provenance: `docs/papers/shared/results/paper6_6_airllm/tinyllama_rtx5060_natural_60_summary.json`; evidence: Natural workload.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for current AirLLM deployments, especially one-shot or low-reuse resources.

## When Native Memory may help

Only investigate workloads where a large immutable resource is reused enough to amortize both reference encoding and layer streaming.

## Limitations

- Warm reuse did not remove request-path overhead
- Beyond-VRAM native CUDA validation remains incomplete

## Research evidence

Current public evidence label: **Natural workload**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Measure disabled, preselected, and routed paths separately
- Attribute reference encoding and per-layer transfer independently

## Production recommendation

Prefer Selected Context; keep Native Memory research-only.
