# vLLM

_Registry reviewed through 2026-09-12; generated from checked-in registries._

**Historical-engine-evidence notice:** Pre-gate Native Memory workload values are retained for audit only. They do not support current exactness, latency, throughput, memory, cache-hit, K/V-copy, or byte-saving claims. Current evidence is limited to the explicitly named post-fix mechanism gates; matched workload economics remain NOT_MEASURED.

## What this engine is for

High-throughput CUDA serving with continuous batching and automatic prefix caching.

## Best PRA deployment today

Selected Context is qualified. Native CUDA correctness and concurrency are promising but matched economics remain pending.

## What PRA adds to this engine

PRA gives vLLM a query-addressed context layer above ordinary
prompt construction. Long-lived documents, tool results, task state, and other
typed resources remain separately addressable; the request receives only the
authorized regions selected for that operation. This reduces visible context
without requiring Native Memory. Deeper native reuse is enabled only where the
table below says it has been measured for this engine.

For vLLM, the practical boundary is: Selected Context is qualified. Native CUDA correctness and concurrency are promising but matched economics remain pending.

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
| Native Memory | 🧪 Candidate (bounded post-fix scheduler-page gate; workload economics NOT_MEASURED) |
| Native Serving | 🧪 Candidate |

**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.

## Architecture

A bounded post-fix scheduler-page alias gate passes exactness, lifecycle, and copy-accounting checks. Earlier connector throughput and recovery counts are quarantined.

```text
application -> typed context -> PRA route/select/materialize
            -> vLLM -> generated response
```

## Requirements and tested boundary

- CUDA-capable NVIDIA GPU
- Supported vLLM release
- PRA gateway for typed transport

## Install and launch

Run these commands in order:

```bash
pra runtime doctor -e vllm
pra runtime inspect Qwen/Qwen3-1.7B -e vllm
pra runtime serve Qwen/Qwen3-1.7B -e vllm
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
pra engines --details vllm
pra evaluate MODEL --engine vllm --dataset DATASET \
  --measurements RESULTS.json -o .pra/runs/engine-evaluation
pra recommend .pra/runs/engine-evaluation
pra report .pra/runs/engine-evaluation --format html
```

## Metrics from the engine paper

These values are imported from the checked-in paper artifacts. They apply to
the named model, workload, hardware, and engine version rather than every deployment.

| Metric | Value | Evidence | Source |
| --- | --- | --- | --- |
| Post-fix scheduler-page alias gate | 7 turns and 14 concurrent borrowers exact; zero K/V copy, H2D, and selected-history re-encoding | Controlled | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/engine_gates/vllm_cuda_scheduler_alias_task02_full7_090_qwen15_v2.json) |
| Matched native economics | NOT_MEASURED | Not measured | [artifact](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md) |

## Metrics and explicit gaps

- **Post-fix scheduler-page alias gate:** 7 turns and 14 concurrent borrowers exact; zero K/V copy, H2D, and selected-history re-encoding  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/engine_gates/vllm_cuda_scheduler_alias_task02_full7_090_qwen15_v2.json`; evidence: Controlled.
- **Matched native economics:** NOT_MEASURED  Provenance: `docs/papers/shared/results/paper4_5_runtime_productization/ENGINE_EVIDENCE_RERUN_AUDIT.md`; evidence: Not measured.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for production throughput today, especially when prefix caching already captures reuse.

## When Native Memory may help

Reconsider after matched cold, hot, warm, APC, transfer, and tail-latency rows are qualified.

## Limitations

- Current CUDA gate is in-process V1 with complete pages and one homogeneous K/V group
- Pre-gate connector throughput is quarantined
- Final HBM, transfer, and tail-latency economics are incomplete

## Research evidence

Current public evidence label: **Controlled**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Inspect the effective gateway and engine capabilities
- Treat a missing native receipt as Selected Context, not silent native execution

## Production recommendation

Deploy Selected Context; keep Native Memory and Native Serving experimental.
