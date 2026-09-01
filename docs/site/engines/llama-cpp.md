# llama.cpp

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Portable local inference across CPU, Metal, CUDA, and other backends.

## Best PRA deployment today

Selected Context through prompts or a compatible server.

## What PRA adds to this engine

PRA gives llama.cpp a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For llama.cpp, the practical boundary is: Selected Context through prompts or a compatible server.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | ✅ Validated |
| Typed PRA Transport | ✅ Validated |
| Native Memory | 🧪 Candidate |
| Native Serving | ⏳ Not measured |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

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

## Install and launch

Run these commands in order:

```bash
pra gateway serve --mode selected-context --backend llama_cpp --backend-url http://127.0.0.1:8080/v1
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
pra engines --details llama-cpp
pra evaluate MODEL --engine llama-cpp --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Selected/full initial prompt tokens | 37.5%-51.7% | Controlled | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |
| Qualified native serving economics | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |

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
