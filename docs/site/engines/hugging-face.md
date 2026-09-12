# Hugging Face

_Registry reviewed through 2026-09-12; generated from checked-in registries._

**Historical-engine-evidence notice:** Pre-gate Native Memory workload values are retained for audit only. They do not support current exactness, latency, throughput, memory, cache-hit, K/V-copy, or byte-saving claims. Current evidence is limited to the explicitly named post-fix mechanism gates; matched workload economics remain NOT_MEASURED.

## What this engine is for

Reference integration for model development, correctness checks, and portable Python experiments.

## Best PRA deployment today

Use Selected Context. Native Memory has a bounded post-fix batch-one Qwen/CUDA mechanism gate, not current workload qualification.

## What PRA adds to this engine

PRA gives Hugging Face a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For Hugging Face, the practical boundary is: Use Selected Context. Native Memory has a bounded post-fix batch-one Qwen/CUDA mechanism gate, not current workload qualification.

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
| Native Serving | ⏳ Not measured |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

## Architecture

The reference runtime supports typed resources and a bounded same-consumer Native Memory gate. Cross-family and workload economics from pre-gate artifacts are quarantined.

```text
application -> typed context -> PRA route/select/materialize
            -> Hugging Face -> generated response
```

## Requirements and tested boundary

- Python 3.10 or newer
- PyTorch and Transformers
- A model adapter supported by the runtime

## Install and launch

Run these commands in order:

```bash
pra runtime doctor -e hf
pra runtime inspect Qwen/Qwen3-1.7B -e hf
pra runtime serve Qwen/Qwen3-1.7B -e hf --storage balanced
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
pra engines --details hugging-face
pra evaluate MODEL --engine hugging-face --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Post-fix same-consumer gate | 7/7 token-exact; max logit delta 0; zero selected-history re-encoding and persistent K/V copy | Controlled | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/engine_gates/hf_fused_sparse_position_task02_full7_090_qwen15_v1.json) |
| Matched workload economics | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md) |

## Metrics and explicit gaps

- **Post-fix same-consumer gate:** 7/7 token-exact; max logit delta 0; zero selected-history re-encoding and persistent K/V copy  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/engine_gates/hf_fused_sparse_position_task02_full7_090_qwen15_v1.json`; evidence: Controlled.
- **Matched workload economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for maximum engine portability, new models, and workloads without repeated immutable evidence.

## When Native Memory may help

Use only to reproduce the named batch-one Qwen/CUDA gate until clean exact-identity workload reruns pass.

## Limitations

- No production scheduler ownership
- Pre-gate cross-family and workload measurements are quarantined
- Current native evidence is limited to the declared batch-one Qwen/CUDA geometry

## Research evidence

Current public evidence label: **Controlled**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Run pra doctor in the same Python environment as the CLI
- Use pra model validate before enabling a native profile

## Production recommendation

Use Selected Context; do not promote Native Memory from the quarantined bundle measurements.
