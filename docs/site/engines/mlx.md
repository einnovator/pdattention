# MLX

_Registry reviewed through 2026-09-12; generated from checked-in registries._

**Historical-engine-evidence notice:** Pre-gate Native Memory workload values are retained for audit only. They do not support current exactness, latency, throughput, memory, cache-hit, K/V-copy, or byte-saving claims. Current evidence is limited to the explicitly named post-fix mechanism gates; matched workload economics remain NOT_MEASURED.

## What this engine is for

Apple-silicon execution and lifecycle research with unified memory.

## Best PRA deployment today

Selected Context is the portable default. Native Memory has a bounded post-fix Qwen3 mechanism gate; workload qualification is pending.

## What PRA adds to this engine

PRA gives MLX a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For MLX, the practical boundary is: Selected Context is the portable default. Native Memory has a bounded post-fix Qwen3 mechanism gate; workload qualification is pending.

## Three kinds of reuse

Selected Context session deduplication is owned by the shared PRA runtime.
Engine-native prefix caching is measured independently. Native semantic
memory is used only when this engine/model/hardware path is qualified.
PRA avoids sending selected context again when it is already active, lets
the inference engine reuse ordinary prefix cache where available, and can
reuse native semantic memory on qualified integrations.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | ✅ Validated |
| Typed PRA Transport | ✅ Validated |
| Native Memory | ✅ Validated (bounded post-fix mechanism gate; workload economics NOT_MEASURED) |
| Native Serving | 🧪 Candidate |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

## Architecture

The post-fix interval-addressed Metal consumer passes its bounded same-consumer and lifecycle gates. Earlier model-scaling and cost measurements are quarantined.

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
- `--mode` / `-m` selects `auto`, `selected-context`, `native-memory`, or
  `native-serving`. Native modes require qualification; `auto` remains
  conservative when incremental economics are not qualified.
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
| Post-fix same-consumer gate | 7/7 token- and final-logit-exact; zero selected-history re-encoding, selection pack bytes, and allocator delta | Controlled | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/engine_gates/mlx_interval_metal_sparse_lifecycle_task02_090_v2.json) |
| Matched workload economics | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md) |
| Reduced consumer-layer profile | CALIBRATION_PENDING | Candidate | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md) |

## Metrics and explicit gaps

- **Post-fix same-consumer gate:** 7/7 token- and final-logit-exact; zero selected-history re-encoding, selection pack bytes, and allocator delta  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/engine_gates/mlx_interval_metal_sparse_lifecycle_task02_090_v2.json`; evidence: Controlled.
- **Matched workload economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md`; evidence: Not measured.
- **Reduced consumer-layer profile:** CALIBRATION_PENDING  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md`; evidence: Candidate.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for one-shot context, unqualified models, or the simplest operational path.

## When Native Memory may help

Use only within the named post-fix mechanism gate until clean exact-identity workload reruns pass.

## Limitations

- Pre-gate model-scaling and workload economics are quarantined
- Reduced consumer-layer profiles did not pass held-out quality gates
- Broad model and serving qualification remains pending

## Research evidence

Current public evidence label: **Controlled**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Keep BALANCED on all eligible layers
- Check model and tokenizer revision before reusing native memory

## Production recommendation

Use Selected Context; keep Native Memory experimental and reduced profiles at CALIBRATION_PENDING.
