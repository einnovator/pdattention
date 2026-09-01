# AirLLM

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Layer-streamed inference for models that exceed accelerator memory.

## Best PRA deployment today

Selected Context; it reduces visible work without adding a separate native reference pass.

## What PRA adds to this engine

PRA gives AirLLM a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For AirLLM, the practical boundary is: Selected Context; it reduces visible work without adding a separate native reference pass.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | ✅ Validated |
| Typed PRA Transport | ✅ Validated |
| Native Memory | 🧪 Research only |
| Native Serving | ⏳ Not measured |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

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

## Install and launch

Run these commands in order:

```bash
pra gateway serve --mode selected-context --backend custom --backend-url http://127.0.0.1:8000/v1
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
pra engines --details airllm
pra evaluate MODEL --engine airllm --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Native/selected TTFT | 2.186x-2.351x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_6_airllm/tinyllama_cuda_natural_summary.json) |
| Native/selected ITL or completion | approximately 1.10x-1.17x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_6_airllm/tinyllama_cuda_natural_summary.json) |
| Mean separate reference encoding | 10.91 seconds | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_6_airllm/tinyllama_rtx5060_natural_60_summary.json) |

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
