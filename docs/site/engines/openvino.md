# OpenVINO

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Intel CPU and integrated-GPU inference through OpenVINO GenAI or OVMS.

## Best PRA deployment today

Selected Context through an ordinary OpenVINO prompt path.

## What PRA adds to this engine

PRA gives OpenVINO a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For OpenVINO, the practical boundary is: Selected Context through an ordinary OpenVINO prompt path.

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
| Native Memory | ⏳ Not qualified |
| Native Serving | ⛔ Not applicable |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

## Architecture

Selected Context reduces input and latency on the measured Intel cohorts. No legitimate detached native-memory seam is qualified.

```text
application -> typed context -> PRA route/select/materialize
            -> OpenVINO -> generated response
```

## Requirements and tested boundary

- OpenVINO GenAI or OVMS
- Supported Intel CPU or GPU
- PRA gateway for typed resources

## Install and launch

Run these commands in order:

```bash
pra doctor
pra gateway serve --mode selected-context --backend custom --backend-url http://127.0.0.1:8000/v1
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
pra engines --details openvino
pra evaluate MODEL --engine openvino --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Full/selected source tokens | 3.53x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_3_openvino/cross_model_summary.json) |
| Full/selected median TTFT | 1.85x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_3_openvino/cross_model_summary.json) |
| Full/selected p95 TTFT | 5.27x | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_3_openvino/cross_model_summary.json) |
| Detached native memory | NOT_APPLICABLE | Not applicable | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |

## Metrics and explicit gaps

- **Full/selected source tokens:** 3.53x  Provenance: `docs/papers/shared/results/paper6_3_openvino/cross_model_summary.json`; evidence: Natural workload.
- **Full/selected median TTFT:** 1.85x  Provenance: `docs/papers/shared/results/paper6_3_openvino/cross_model_summary.json`; evidence: Natural workload.
- **Full/selected p95 TTFT:** 5.27x  Provenance: `docs/papers/shared/results/paper6_3_openvino/cross_model_summary.json`; evidence: Natural workload.
- **Detached native memory:** NOT_APPLICABLE  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not applicable.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for Intel deployments where token reduction improves prefill and process work.

## When Native Memory may help

Do not plan around it until OpenVINO exposes and qualifies a detached non-prefix memory seam.

## Limitations

- Quality remains selector and workload dependent
- No qualified Native Memory path

## Research evidence

Current public evidence label: **Natural workload**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Freeze selector output when comparing full and selected prompts
- Record device, model conversion, and OpenVINO versions

## Production recommendation

Use Selected Context and report quality together with token and latency reduction.
