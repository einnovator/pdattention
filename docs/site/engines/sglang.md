# SGLang

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Structured generation and cache-aware serving with RadixAttention and hierarchical cache facilities.

## Best PRA deployment today

Selected Context is the default. A companion native mechanism is validated, while distributed scheduler economics remain open.

## What PRA adds to this engine

PRA gives SGLang a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For SGLang, the practical boundary is: Selected Context is the default. A companion native mechanism is validated, while distributed scheduler economics remain open.

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
| Native Serving | 🧪 Candidate |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

## Architecture

Native resources are isolated from ordinary sequential cache state. Full distributed placement, affinity, and concurrent tier economics are still candidates.

```text
application -> typed context -> PRA route/select/materialize
            -> SGLang -> generated response
```

## Requirements and tested boundary

- Supported SGLang server or companion runner
- PRA gateway
- Explicit tenant and session scope

## Install and launch

Run these commands in order:

```bash
pra runtime doctor -e sglang
pra runtime inspect Qwen/Qwen3-1.7B -e sglang
pra runtime serve Qwen/Qwen3-1.7B -e sglang
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
pra engines --details sglang
pra evaluate MODEL --engine sglang --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Matched quality and lifecycle cohorts | Available in registry | Natural workload | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper6_1_sglang/expanded_matched_e0_e2_qasper.json) |
| Distributed HiCache economics | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/pra_product_matrix_v2.json) |

## Metrics and explicit gaps

- **Matched quality and lifecycle cohorts:** Available in registry  Provenance: `docs/papers/shared/results/paper6_1_sglang/expanded_matched_e0_e2_qasper.json`; evidence: Natural workload.
- **Distributed HiCache economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it when ordinary Radix/prefix behavior is sufficient or native placement has not been qualified.

## When Native Memory may help

Consider it for immutable shared resources under the companion path and explicit isolation tests.

## Limitations

- Distributed HiCache placement is not fully lifecycle-managed
- Concurrent cold/warm tail curves remain incomplete

## Research evidence

Current public evidence label: **Serving**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Never return selected native detail to the ordinary Radix cache pool
- Verify cleanup and one-copy attachment per request

## Production recommendation

Use Selected Context; treat Native Serving as a measured deployment project, not a flag.
