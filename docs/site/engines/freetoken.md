# FreeToken

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Logical context and bandwidth coordination research across compatible endpoints.

## Best PRA deployment today

Selected Context or Typed PRA Transport at the application boundary.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Candidate |
| Typed PRA Transport | Candidate |
| Native Memory | Not measured |
| Native Serving | Not measured |

## Architecture

Current evidence concerns coordination and bandwidth, not live native-memory serving.

```text
application -> typed context -> PRA route/select/materialize
            -> FreeToken -> generated response
```

## Requirements and tested boundary

- Compatible endpoint
- Explicit transport contract
- Independent quality and authorization checks

## Quickstart

```bash
pra gateway serve --mode selected-context --backend freetoken --backend-url http://127.0.0.1:8000/v1
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Live native qualification | NOT_MEASURED | Not measured |
| Serving economics | NOT_MEASURED | Not measured |

## Metrics and explicit gaps

- **Live native qualification:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.
- **Serving economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it only when the endpoint contract and selected-text baseline are reproducible.

## When Native Memory may help

Not currently recommended.

## Limitations

- No live native-serving claim
- Coordination metrics do not establish model-quality or serving gains

## Research evidence

Current public evidence label: **Controlled**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Preserve NOT_MEASURED instead of converting it to zero
- Separate network token reduction from infrastructure cost

## Production recommendation

Treat as research evidence, not a qualified engine deployment.
