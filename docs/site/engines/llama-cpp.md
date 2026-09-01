# llama.cpp

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Portable local inference across CPU, Metal, CUDA, and other backends.

## Best PRA deployment today

Selected Context through prompts or a compatible server.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Candidate |
| Native Serving | Not measured |

## Architecture

Ordinary slot and session cache state is sequential prefix state, not automatically PRA Native Memory. A separate experimental sequence-attachment seam requires explicit qualification.

```text
application -> typed context -> PRA route/select/materialize
            -> llama.cpp -> generated response
```

## Requirements and tested boundary

- llama.cpp server or local runtime
- A compatible GGUF model
- PRA gateway for typed transport

## Quickstart

```bash
pra gateway serve --mode selected-context --backend llama_cpp --backend-url http://127.0.0.1:8080/v1
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Selected/full initial prompt tokens | 37.5%-51.7% | Controlled |
| Qualified native serving economics | NOT_MEASURED | Not measured |

## Metrics and explicit gaps

- **Selected/full initial prompt tokens:** 37.5%-51.7%  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Controlled.
- **Qualified native serving economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for portable local deployments and unmodified upstream builds.

## When Native Memory may help

Only with a pinned PRA-aware backend receipt and exact isolation, parity, and lifecycle tests.

## Limitations

- Prefix/session cache reuse is not detached semantic memory
- Native attachment is not part of the stock public runtime

## Research evidence

Current public evidence label: **Controlled**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Do not label slot reuse as Native Memory
- Require model fingerprint and backend revision in any native capability receipt

## Production recommendation

Use Selected Context with stock llama.cpp.
