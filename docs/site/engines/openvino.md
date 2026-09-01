# OpenVINO

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Intel CPU and integrated-GPU inference through OpenVINO GenAI or OVMS.

## Best PRA deployment today

Selected Context through an ordinary OpenVINO prompt path.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Not qualified |
| Native Serving | Not applicable |

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

## Quickstart

```bash
pra doctor
pra gateway serve --mode selected-context --backend custom --backend-url http://127.0.0.1:8000/v1
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Full/selected source tokens | 3.53x | Natural workload |
| Full/selected median TTFT | 1.85x | Natural workload |
| Full/selected p95 TTFT | 5.27x | Natural workload |
| Detached native memory | NOT_APPLICABLE | Not applicable |

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
