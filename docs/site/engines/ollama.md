# Ollama

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Local model packaging and a simple HTTP serving surface.

## Best PRA deployment today

Selected Context through the gateway and Ollama API.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Candidate |
| Native Serving | Not measured |

## Architecture

Keep-alive can improve reuse, but it remains ordinary model lifecycle behavior. Deeper native use requires a validated PRA-aware backend capability handshake.

```text
application -> typed context -> PRA route/select/materialize
            -> Ollama -> generated response
```

## Requirements and tested boundary

- Ollama server
- Installed model
- PRA gateway for typed resources

## Quickstart

```bash
pra doctor
pra gateway serve --mode selected-context --backend ollama --backend-url http://127.0.0.1:11434
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Selected prompt latency | Improved on current 14B HotpotQA/QASPER cohorts; quality is workload dependent | Natural workload |
| Stock backend native memory | NOT_MEASURED | Not measured |

## Metrics and explicit gaps

- **Selected prompt latency:** Improved on current 14B HotpotQA/QASPER cohorts; quality is workload dependent  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Natural workload.
- **Stock backend native memory:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for stock Ollama installations and broad model compatibility.

## When Native Memory may help

Only when the backend returns a validated protocol, model fingerprint, mechanism list, and lifecycle receipt.

## Limitations

- Keep-alive is not detached semantic memory
- Quality gains depend on selection quality

## Research evidence

Current public evidence label: **Natural workload**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Treat malformed or stale capability receipts as Selected Context
- Invalidate native receipts on model switch or unload

## Production recommendation

Use Selected Context; deeper integration is opt-in and receipt-gated.
