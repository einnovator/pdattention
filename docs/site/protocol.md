# Agent, Gateway, and Engine Protocol

PRA extends OpenAI-compatible chat requests without replacing their message
protocol. The conversation remains an ordinary sequence of system, user,
assistant, and provider-required tool messages. Large or reusable context is a
separate stream of typed logical resources.

```text
AgentTurnContext
  messages -------------------------- OpenAI conversation spine
  records/tools/skills/tasks -------- detached PRA resources
                         |
                         v
                  capability negotiation
                 /                      \
          ordinary endpoint        PRA-aware endpoint
          selected text          typed full/delta records
```

## Capability handshake

At launch or reconnect, a client requests:

```http
GET /v1/pra/capabilities
```

The response separates gateway features, engine features, and effective
end-to-end features. Clients check capabilities such as `logical_refs`,
`typed_records`, `task_metadata`, `resource_delta`, `session_state`, and
`incremental_messages`; they do not branch on engine names.

A reachable endpoint returning an unsupported-route status can be treated as an
ordinary OpenAI endpoint. An unreachable endpoint is an error, not permission to
silently downgrade.

## Transport policy

Agent profiles choose `auto`, `pra`, or `text` transport:

```yaml
context:
  transport: auto
  allow_text_fallback: true
  require:
    - logical_refs
    - typed_records
```

Resolved behavior is one of:

| Public behavior | Messages | Detached resources |
| --- | --- | --- |
| Pass through | Full OpenAI messages | None |
| Selected Context | Full OpenAI messages | Deterministically rendered selected records |
| Typed full transport | Full OpenAI messages | Full logical resource inventory |
| Typed delta transport | Incremental messages | Add, update, remove, or unchanged operations |

Native K/V is not a transport mode. Raw tensors and physical page identifiers
never cross this wire.

## Request envelope

PRA-aware requests retain ordinary top-level chat fields:

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

Ordinary endpoints receive no `pra` field. Credentials stay in HTTP headers or
local provider configuration and are prohibited from resource, provenance, and
trace metadata.

## Record projection

`context_record_to_wire_resource()` preserves:

- record ID, type, URI, version, and source fingerprint;
- provenance and authorization scope;
- task ID and status;
- available, initial, and selected views;
- tenant/session binding and explicit shareability.

The representations remain distinct:

| Representation | Owner | Purpose |
| --- | --- | --- |
| `ContextRecord` | Agent/runtime | Semantic typed record and named views |
| `PRAWireResource` | Transport | Portable logical network projection |
| `BackingRecord` | Storage | Reconstructible authoritative detail |
| `PRAStorageEntry` | Engine/storage | Model-specific derived native residency |

## Sessions, deltas, and resynchronization

The transport caches acknowledged message and resource inventories per session.
Endpoint changes, reconnects, engine restarts, protocol mismatch, or explicit
refresh invalidate that state. The next request sends full messages and a full
resource inventory before deltas resume.

Resource identity changes when version, source fingerprint, authorization, task
ownership, or shareability changes. Storage deduplication never grants access
across tenants or sessions.

## Responses and streaming

Non-streaming responses remain OpenAI-compatible and may add a `pra` object with
selected resource IDs, materialized token counts, native-memory status, and a
trace ID. Streaming uses ordinary `chat.completion.chunk` content events.
Additive PRA trace events carry no content choices, so conventional clients can
continue reading text deltas.

## Fallback rules

- Fallback must be explicitly permitted by the request or profile.
- Selected Context fallback must preserve identity labels and authorization.
- A request requiring Native Memory fails if the engine cannot provide it.
- Capability loss invalidates cached native receipts.
- The trace records the selected public behavior and any fallback reason.

The legacy research codes that map to these public behaviors are documented
only in [Research / Evidence](research/index.md).
