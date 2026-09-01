# TensorRT-LLM

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Optimized NVIDIA inference using compiled engines and paged serving primitives.

## Best PRA deployment today

Selected Context through the compatible request path.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Candidate |
| Typed PRA Transport | Validated |
| Native Memory | Not measured |
| Native Serving | Not measured |

## Architecture

The transport and runtime boundary are defined, but native quality and economics are not yet qualified.

```text
application -> typed context -> PRA route/select/materialize
            -> TensorRT-LLM -> generated response
```

## Requirements and tested boundary

- NVIDIA GPU
- TensorRT-LLM engine built for the model
- Compatible gateway endpoint

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
| Matched natural serving benchmark | NOT_MEASURED | Not measured |

## Metrics and explicit gaps

- **Matched natural serving benchmark:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it when a conventional TensorRT-LLM engine is already operational.

## When Native Memory may help

Wait for a versioned native attention seam and matched quality/economic evidence.

## Limitations

- No qualified native-memory implementation in the product runtime
- Engine builds are model and hardware specific

## Research evidence

Current public evidence label: **Candidate**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Validate the ordinary endpoint before adding typed transport
- Do not infer native support from paged-cache support alone

## Production recommendation

Use Selected Context; native modes are not yet product-qualified.
