# MLX

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Apple-silicon execution and lifecycle research with unified memory.

## Best PRA deployment today

Selected Context is the portable default. Native Memory is validated with all eligible consumer layers for measured models.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Validated |
| Native Serving | Candidate |

## Architecture

Synchronized larger-model measurements approach Selected Context cost parity as model scale grows; they do not establish a universal native speedup.

```text
application -> typed context -> PRA route/select/materialize
            -> MLX -> generated response
```

## Requirements and tested boundary

- Apple silicon
- MLX and mlx-lm
- A supported Qwen or reference-compatible model

## Quickstart

```bash
pra runtime doctor -e mlx
pra runtime inspect mlx-community/Qwen3-4B-4bit -e mlx
pra runtime serve mlx-community/Qwen3-4B-4bit -e mlx --storage balanced
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Warm native/selected cost, 4B | 1.035x | Natural workload |
| Warm native/selected cost, 8B | 1.015x | Natural workload |
| Warm native/selected cost, 14B | 0.980x; interval includes parity | Natural workload |
| Reduced consumer-layer profile | CALIBRATION_PENDING | Candidate |

## Metrics and explicit gaps

- **Warm native/selected cost, 4B:** 1.035x  Provenance: `docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json`; evidence: Natural workload.
- **Warm native/selected cost, 8B:** 1.015x  Provenance: `docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json`; evidence: Natural workload.
- **Warm native/selected cost, 14B:** 0.980x; interval includes parity  Provenance: `docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json`; evidence: Natural workload.
- **Reduced consumer-layer profile:** CALIBRATION_PENDING  Provenance: `docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json`; evidence: Candidate.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for one-shot context, unqualified models, or the simplest operational path.

## When Native Memory may help

Consider it for multi-query reuse over immutable resources when the exact model profile has passed quality gates.

## Limitations

- The segmented implementation is not fully fused
- Reduced consumer-layer profiles did not pass held-out quality gates
- The 16 GiB M5 cannot load the measured 32B configuration

## Research evidence

Current public evidence label: **Natural workload**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Keep BALANCED on all eligible layers
- Check model and tokenizer revision before reusing native memory

## Production recommendation

Use BALANCED only where measured; keep reduced profiles at CALIBRATION_PENDING.
