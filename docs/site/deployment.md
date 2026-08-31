# Execution Policies and Deployment

PRA selection, K/V materialization, and process placement are separate choices. The
SDK keeps them separate so a request cannot silently change semantics while moving
between a local model and a remote engine.

## Execution policy

`PRAExecutionPolicy` has four independent axes:

| Axis | Initial values | Meaning |
| --- | --- | --- |
| Selection stage | `request`, `phase`, `token` | When logical identities may change |
| Layer scope | `shared`, `per_layer` | Whether layers reuse identities or route independently |
| Materialization scope | `request`, `phase`, `layer`, `token` | Lifetime of a physical payload |
| Residency | `keep`, `layer_lifetime` | Whether a payload remains resident for its lifetime |

The global default is `request/shared/request/keep`. Request overrides take precedence
over model defaults, which take precedence over the global default. Unsupported
combinations raise an error; the SDK does not silently choose a cheaper mode.

```python
from pra_hf import PRAForCausalLM

model = PRAForCausalLM.from_pretrained(model_id)
model.set_execution_policy(selection_layer_scope="per_layer")
result = model.generate(
    prompt,
    pra_policy={
        "selection_stage": "token",
        "selection_layer_scope": "shared",
        "materialization_scope": "token",
        "routing_layer_policy": "first_pra_layer",
    },
    return_details=True,
)
print(result.stats["pra_execution"])
```

The HF reference backend implements request/shared, request/per-layer,
token/shared, and token/per-layer. In token/shared mode, layers before the routing
layer receive no current-token memory. The routing layer selects once; that logical
identity set is mapped to each later layer's independently encoded native K/V.
Phase-level execution is currently rejected because it needs a cache-aware
prefill-to-decode handoff that the reference `generate()` path does not yet expose.

## Three components

1. **Agent or harness:** owns task/session meaning, record structure, provenance, and authorization references.
2. **PRA gateway:** owns protocol translation, logical resource IDs, explicit downgrade decisions, and transport traces.
3. **Inference engine:** owns tensor layout, K/V blocks, device placement, attention, and scheduling.

Raw K/V does not cross the normal harness/gateway wire boundary. `PRAWireRequest`
contains JSON-serializable messages, stable resource IDs, budgets, query facets,
policy hints, and requested capabilities.

The agent first creates an `AgentTurnContext`: OpenAI-style conversational
messages remain the pretrained chat spine, while documents, task results, tools,
and skills remain detached `ContextRecord` objects. AUTO transport projects them
to `PRAWireResource` when the immediate endpoint advertises typed records. Only
an ordinary endpoint receives the canonical text rendering.

## Runtime providers

`pra runtime serve MODEL -e ENGINE` is the canonical engine launch path. Click
commands delegate to `RuntimeManager`, which resolves a `RuntimeProvider` and
translates the common model/revision/PRA profile/session contract into the
upstream engine's supported launcher. Repeated `--engine-arg KEY=VALUE` options
remain an escape hatch rather than cloning every upstream CLI.

```text
pra runtime serve
        |
 RuntimeProvider
   /    |     |     |       \
 HF   vLLM  SGLang  MLX   AirLLM
```

The gateway remains optional and separate:

```text
agent -> gateway -> runtime -> engine
```

`runtime inspect` reports static provider declarations, while `runtime doctor`
checks installed dependencies and remote readiness. The vLLM provider remains
E0 because its safe V1 metadata boundary is not yet wired through generation.
The SGLang and MLX providers expose the companion papers' measured E2 native
mechanism integrations without claiming E3 scheduler participation.

AirLLM is embedded-only. It defaults to selected-text E0 on every platform.
Its HF-backed CUDA path may opt into `native_hf_pra=true` after the adapter is
attached; AirLLM's separate macOS/MLX implementation remains E0. This platform
check prevents an installed package from being mistaken for native support.

## Gateway modes

| Mode | Input | Engine | Behavior |
| --- | --- | --- | --- |
| G00 | ordinary | ordinary | pass through structured messages |
| G10 | PRA-aware | ordinary | selected resources become labeled text context |
| G01 | ordinary | PRA-aware | infer typed resources from system/tool records |
| G11 | PRA-aware | PRA-aware | preserve logical PRA semantics end to end |

G10 is **PRA control-plane / text materialization**, not native PRA. A downgrade is
allowed only when the request opts into text fallback, and the trace names it.

```bash
pra gateway serve \
  --host 0.0.0.0 --port 8080 \
  --mode G10 \
  --backend sglang \
  --backend-url http://localhost:30000
```

The gateway exposes `GET /health`, `GET /v1/pra/capabilities`,
`POST /v1/pra/generate`, and `POST /v1/chat/completions`. It is currently
streaming when its engine adapter supports streaming. HTTP content events use
OpenAI `chat.completion.chunk` deltas; additive PRA-only chunks carry trace
metadata with an empty `choices` list. Capability responses distinguish gateway,
engine, and effective end-to-end features. See [Agent/Gateway Protocol](protocol.md).

## Engine levels

| Level | Meaning | Current status |
| --- | --- | --- |
| E0 | selected-text compatibility through an ordinary model path | implemented for OpenAI-compatible HTTP |
| E1 | logical PRA identity, metadata, deltas, authorization, and fallback | supported by the typed gateway contract |
| E2 | detached/native non-prefix PRA attention | HF reference; companion vLLM/SGLang/MLX mechanisms; AirLLM/HF mechanism smoke |
| E3 | scheduler-owned placement, prefetch, eviction, sharing, and batching | candidates only; not generally qualified |

SGLang documents an OpenAI-compatible server and a model gateway. The companion
Paper 6.1 integration exercises its MLX runner at E2, while the generic remote
provider remains usable at E0
([SGLang quickstart](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/quickstart.mdx),
[attention backends](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)).
FreeToken documents OpenAI and Anthropic endpoints, which makes E0 transport
feasible; native PRA cache integration remains unimplemented
([FreeToken quickstart](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)).

## Security and ownership

Retrieval relevance never grants source access or tool permission. The wire parser
rejects cross-tenant resources and credential-like request metadata. Engine handles
must remain tenant/session scoped, cache presence is not authorization, and tool side
effects continue through the independent host authorization boundary.

## Agent integrations

DeepSeek Harness and Pi can use the OpenAI-compatible gateway without native model
support. `DeepSeekHarnessPRAAdapter` consumes durable `tool/result`,
`session/reference`, and attachment events. `PiCodingAgentPRAAdapter` consumes the
documented `tool_execution_end` RPC/extension event and completed `toolResult` or
`bashExecution` messages. Both adapters deduplicate stable event identities, retain
typed provenance plus `session_id` and `task_id`, and emit a tensor-free
`PRAWireRequest`.

```python
from pra_hf import PiCodingAgentPRAAdapter, PRAAgentPluginConfig

bridge = PiCodingAgentPRAAdapter(
    PRAAgentPluginConfig("qwen", allow_text_fallback=True),
    session_id="pi-session",
    task_id="inspect-build",
)
bridge.ingest_event(pi_rpc_event)
result = bridge.generate(gateway, [{"role": "user", "content": "Continue"}])
```

The adapters target the public event seams documented by
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)
and [Pi](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md).
They are Python/RPC bridges rather than patches carried inside those upstream
repositories. Ten five-seed contract cases verify typed identity, deduplication,
task metadata, explicit G10 fallback, and absence of a native-K/V claim. They do not
measure either upstream agent end to end.

## Task acquisition and frozen native geometry

Model-generated JSON, bounded Markdown, and online task operations are parsed as
untrusted proposals. `PRARuntime.apply_task_operations()` validates them against a
copy of the durable task graph before replaying accepted events through the session
service. Adaptive task scope records why it widened and can move from structural to
related and finally session scope when metadata is incomplete.

`freeze_native_selection()`, `plan_native_materialization()`, and
`generate_with_native_plan()` separate routing identity from physical span width.
Expansion is symmetric but record bounded, overlapping intervals are merged, and
every consuming layer receives its own native projected K/V for the same logical
source interval. This is the SDK counterpart of Paper 8's consumption diagnostic;
it does not convert a negative frozen-consumption result into a serving claim.

Storage policy is resolved before provider launch. Use `--storage` for the
`memory`, `balanced`, `persistent`, or `minimal` profile, or
`--storage-config PATH` for record/task-aware YAML. Providers share semantic
`HOT/WARM/COLD/SOURCE` decisions; only their physical HOT representation and
promotion transport differ. See [Storage lifecycle](storage.md).
