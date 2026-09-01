# TensorRT-LLM

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Optimized NVIDIA inference using compiled engines and paged serving primitives.

## Best PRA deployment today

Selected Context through the compatible request path.

## What PRA adds to this engine

PRA gives TensorRT-LLM a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For TensorRT-LLM, the practical boundary is: Selected Context through the compatible request path.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | 🧪 Candidate |
| Typed PRA Transport | ✅ Validated |
| Native Memory | ⏳ Not measured |
| Native Serving | ⏳ Not measured |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

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
pra engines --details tensorrt-llm
pra evaluate MODEL --engine tensorrt-llm --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Matched natural serving benchmark | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |

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
