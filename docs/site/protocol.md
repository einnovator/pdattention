# Agent/Gateway/Engine Protocol

PRA extends the OpenAI chat-completions request without replacing its message
protocol. The live conversation remains an ordinary sequence of system, user,
assistant, and provider-required tool messages. Large or reusable context is a
separate stream of typed logical resources.

```text
AgentTurnContext
  messages ─────────────────────────────── OpenAI conversation spine
  records/tools/skills/tasks ───────────── detached PRA resources
                         │
                         ▼
                  transport negotiation
                 /                     \
           ordinary endpoint       PRA endpoint
              TEXT                 PRA_FULL/DELTA
```

## Capability Handshake

At launch or session reconnect, an agent requests:

```http
GET /v1/pra/capabilities
```

A PRA endpoint returns protocol version `1`, its endpoint type, separate
gateway and engine capabilities, and the effective end-to-end features. A
reachable endpoint returning `404`, `405`, or `501` is treated as ordinary
OpenAI. An unreachable endpoint is an error, not evidence that text fallback
is appropriate.

Capability decisions are feature-based. The agent checks `logical_refs`,
`typed_records`, `task_metadata`, `resource_delta`, `session_state`, and
`incremental_messages`; it does not branch on names such as vLLM or SGLang.

## Transport Policy

Agent profiles use:

```yaml
context:
  transport: auto       # auto | pra | text
  allow_text_fallback: true
  require:
    - logical_refs
    - typed_records
```

`auto` chooses typed transport when the immediate endpoint accepts it and text
otherwise. `pra` requires typed transport unless fallback is explicitly
enabled. `text` is a reproducible compatibility and experimental baseline.

The resolved wire modes are:

| Mode | Messages | Detached resources |
|---|---|---|
| `TEXT` | OpenAI messages | Canonical `PRA_RECORD` text rendering |
| `PRA_FULL` | OpenAI messages | Full logical resource inventory |
| `PRA_DELTA` | Incremental messages | `ADD`, `UPDATE`, `REMOVE`, or body-free `UNCHANGED` operations |

Native K/V is not an agent transport mode. It is an engine capability. Raw K/V
and physical page identifiers never cross this wire.

CLI overrides are independent of `-P/--pra`:

```bash
pra agent chat -p work --context-transport auto
pra agent run -p work --context-transport pra --no-text-fallback "Continue task"
pra agent run -p baseline --context-transport text "Continue task"
```

## OpenAI-Compatible Envelope

PRA-aware requests retain the normal top-level chat fields:

```json
{
  "model": "model-id",
  "messages": [{"role": "user", "content": "Continue the analysis"}],
  "stream": false,
  "pra": {
    "protocol_version": "1",
    "tenant_id": "tenant-a",
    "session_id": "session-a",
    "task_id": "task-1",
    "resources": [],
    "resource_ops": [],
    "budget": {"max_resources": 8, "max_selected_tokens": 2048},
    "pra_policy": {}
  }
}
```

Ordinary OpenAI endpoints receive no `pra` field. Credentials remain in HTTP
headers or local credential configuration and are prohibited from PRA request,
resource, provenance, and trace metadata.

## Record Projection

`context_record_to_wire_resource()` is the canonical projection. It preserves:

- record ID, type, URI, version, and source fingerprint;
- provenance and authorization scope;
- task ID/status;
- available, initial, and selected named views;
- tenant/session binding and explicit shareability.

It sends the selected logical body only when required. An unchanged resource
uses a body-free operation after the gateway or engine has acknowledged its
inventory.

Four representations remain distinct:

| Representation | Owner | Purpose |
|---|---|---|
| `ContextRecord` | Agent/runtime | Semantic typed record and named views |
| `PRAWireResource` | Transport | Portable logical network projection |
| `BackingRecord` | Storage | Reconstructible SOURCE descriptor |
| `PRAStorageEntry` | Engine/storage | Model-specific native-K/V residency |

The mappings are `ContextRecord -> PRAWireResource`, source content to
`BackingRecord`, and source plus model encoding to `PRAStorageEntry` and native
K/V.

## Gateway Modes

- `G00` passes through a compatible request.
- `G10` accepts typed resources from the agent and performs deterministic text
  injection for an E0 engine.
- `G11` preserves resources, task/session metadata, and deltas for a PRA-aware
  engine.
- `G01` cautiously recognizes supported typed ordinary traffic and upgrades it.

AUTO always negotiates with the immediate endpoint. Therefore a G10 gateway
receives typed resources even though its downstream engine is E0. Protocol
downgrade belongs to the gateway, not the agent.

## Sessions, Deltas, and Resynchronization

The transport caches capabilities and acknowledged message/resource inventory
per session. Endpoint changes, reconnects, engine restarts, protocol mismatch,
or explicit refresh invalidate this state. The next request sends full messages
and an `ADD` inventory before delta mode resumes.

Resource identity changes when version, source fingerprint, authorization,
task ownership, or shareability changes. Storage deduplication never grants
authorization across tenants or sessions.

Debug traces expose negotiated mode, capability source, fallback, history mode,
delta counts, selected IDs, native-K/V status, and engine integration level.
Bodies and credentials are excluded.

## Responses and Streaming

Non-streaming responses remain OpenAI-compatible and may add:

```json
"pra": {
  "selected_resource_ids": ["record-1"],
  "materialized_tokens": 128,
  "native_kv": true,
  "trace_id": "..."
}
```

Streaming uses ordinary `chat.completion.chunk` events. PRA trace metadata is
carried in additive chunks with an empty `choices` list, so ordinary OpenAI
consumers can continue reading content deltas.
