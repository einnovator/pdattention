# Hugging Face

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Reference integration for model development, correctness checks, and portable Python experiments.

## Best PRA deployment today

Use Selected Context for ordinary pipelines. Use Native Memory when validating model-level PRA behavior or a measured model profile.

## What PRA adds to this engine

PRA gives Hugging Face a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For Hugging Face, the practical boundary is: Use Selected Context for ordinary pipelines. Use Native Memory when validating model-level PRA behavior or a measured model profile.

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
| Native Memory | ✅ Validated |
| Native Serving | ⏳ Not measured |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

## Architecture

The reference runtime supports typed resources and layer-specific native memory, but it is not a scheduler-managed serving system.

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
| Natural serving tails | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |
| Reference mechanism | Validated across Qwen, Llama, and Gemma adapters | Model-backed | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |

## Metrics and explicit gaps

- **Natural serving tails:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.
- **Reference mechanism:** Validated across Qwen, Llama, and Gemma adapters  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Model-backed.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for maximum engine portability, new models, and workloads without repeated immutable evidence.

## When Native Memory may help

Consider it for repeated selected resources after model-specific parity and quality gates pass.

## Limitations

- No production scheduler ownership
- Profile qualification remains model and workload specific

## Research evidence

Current public evidence label: **Model-backed**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Run pra doctor in the same Python environment as the CLI
- Use pra model validate before enabling a native profile

## Production recommendation

Start with Selected Context and promote Native Memory only from a measured bundle.
