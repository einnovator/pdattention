# FreeToken

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Logical context and bandwidth coordination research across compatible endpoints.

## Best PRA deployment today

Selected Context or Typed PRA Transport at the application boundary.

## What PRA adds to this engine

PRA gives FreeToken a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For FreeToken, the practical boundary is: Selected Context or Typed PRA Transport at the application boundary.

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
| Selected Context | 🧪 Candidate |
| Typed PRA Transport | 🧪 Candidate |
| Native Memory | ⏳ Not measured |
| Native Serving | ⏳ Not measured |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

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

## Install and launch

Run these commands in order:

```bash
pra gateway serve --mode selected-context --backend freetoken --backend-url http://127.0.0.1:8000/v1
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
pra engines --details freetoken
pra evaluate MODEL --engine freetoken --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Live native qualification | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |
| Serving economics | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |

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
