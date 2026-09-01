# MLX

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Apple-silicon execution and lifecycle research with unified memory.

## Best PRA deployment today

Selected Context is the portable default. Native Memory is validated with all eligible consumer layers for measured models.

## What PRA adds to this engine

PRA gives MLX a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For MLX, the practical boundary is: Selected Context is the portable default. Native Memory is validated with all eligible consumer layers for measured models.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | ✅ Validated |
| Typed PRA Transport | ✅ Validated |
| Native Memory | ✅ Validated |
| Native Serving | 🧪 Candidate |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

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

## Install and launch

Run these commands in order:

```bash
pra runtime doctor -e mlx
pra runtime inspect mlx-community/Qwen3-4B-4bit -e mlx
pra runtime serve mlx-community/Qwen3-4B-4bit -e mlx --storage balanced
```


### Command options

- `--engine` / `-e` selects the runtime provider used for inspection or launch.
- `--mode selected-context` renders the frozen selected evidence as ordinary
  input. `--mode native-memory` requests a qualified detached-memory path.
  `--mode auto` remains conservative when economics are not qualified.
- `--profile recommended` selects the current qualified model profile; it
  does not promote smoke-only consumer-layer candidates.
- `--storage memory|balanced|persistent|minimal` controls native-resource
  lifecycle when the selected engine exposes it.
- `--backend` names a gateway adapter; `--backend-url` is the existing
  OpenAI-compatible endpoint. The gateway does not own that engine process.
- `--measurements RESULTS.json` imports selector-frozen quality, latency,
  memory, and lifecycle results into `pra evaluate`.

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

### Qualify this exact deployment

```bash
pra engines --details mlx
pra evaluate MODEL --engine mlx --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Warm native/selected cost, 4B | 1.035x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json) |
| Warm native/selected cost, 8B | 1.015x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json) |
| Warm native/selected cost, 14B | 0.980x; interval includes parity | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json) |
| Reduced consumer-layer profile | CALIBRATION_PENDING | Candidate | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_2_mlx/model_consumer_scaling_m5/m5_corrected/summary/model_consumer_scaling_summary.json) |

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
