# SGLang

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Structured generation and cache-aware serving with RadixAttention and hierarchical cache facilities.

## Best PRA deployment today

Selected Context is the default. A companion native mechanism is validated, while distributed scheduler economics remain open.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Validated |
| Native Serving | Candidate |

## Architecture

Native resources are isolated from ordinary sequential cache state. Full distributed placement, affinity, and concurrent tier economics are still candidates.

```text
application -> typed context -> PRA route/select/materialize
            -> SGLang -> generated response
```

## Requirements and tested boundary

- Supported SGLang server or companion runner
- PRA gateway
- Explicit tenant and session scope

## Quickstart

```bash
pra runtime doctor -e sglang
pra runtime inspect Qwen/Qwen3-1.7B -e sglang
pra runtime serve Qwen/Qwen3-1.7B -e sglang
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Matched quality and lifecycle cohorts | Available in registry | Natural workload |
| Distributed HiCache economics | NOT_MEASURED | Not measured |

## Metrics and explicit gaps

- **Matched quality and lifecycle cohorts:** Available in registry  Provenance: `docs/papers/shared/results/paper6_1_sglang/expanded_matched_e0_e2_qasper.json`; evidence: Natural workload.
- **Distributed HiCache economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it when ordinary Radix/prefix behavior is sufficient or native placement has not been qualified.

## When Native Memory may help

Consider it for immutable shared resources under the companion path and explicit isolation tests.

## Limitations

- Distributed HiCache placement is not fully lifecycle-managed
- Concurrent cold/warm tail curves remain incomplete

## Research evidence

Current public evidence label: **Serving**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Never return selected native detail to the ordinary Radix cache pool
- Verify cleanup and one-copy attachment per request

## Production recommendation

Use Selected Context; treat Native Serving as a measured deployment project, not a flag.
