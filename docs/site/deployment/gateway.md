# Gateway Deployment

The gateway lets an OpenAI-compatible application use typed PRA context without
changing the model endpoint protocol. It negotiates capabilities, preserves
resource identity, and owns explicit fallback.

## What the gateway does

The gateway is the coordination boundary between an agent or application and a
model engine. It does not run attention, own model-native K/V, or execute tools.
It turns one logical request into the safest request that the configured engine
can actually execute.

For each request, the gateway performs this sequence:

1. **Accept and normalize the request.** It accepts ordinary OpenAI-compatible
   chat requests and the PRA request envelope. The envelope can carry messages,
   tool schemas, typed resources, a task identifier, tenant and session
   identity, resource operations, selection budgets, and engine hints.
2. **Resolve session state.** It combines the incoming turn with the canonical
   message and resource state for that tenant, session, and model. A capable
   backend receives only message and resource deltas; a stateless backend
   receives the reconstructed state it needs.
3. **Plan realization.** A visibility ledger reconciled against the actual
   serialized model context classifies each selected resource as already
   visible, natively available, newly materialized, or invalid. Selected Context
   injects only the resources that are not already active.
4. **Protect resource boundaries.** Resource ownership is checked against the
   request tenant. Credential-like fields are rejected from PRA metadata, and
   resource authorization scope remains attached during transport. Relevance or
   cache residency never grants access.
5. **Negotiate backend capabilities.** The gateway determines whether the
   backend understands typed records, resource deltas, session state, streaming,
   cache affinity, and Native Memory. Required capabilities either remain intact,
   use an explicitly allowed fallback, or cause a clear request error.
6. **Choose the effective transport.** Depending on the configured mode, it can
   pass the request through, convert selected resources into visible context,
   infer typed resources from compatible message metadata, or preserve typed
   resources for a PRA-aware engine.
7. **Apply context budgets.** In `selected-context` mode, the reference gateway
   limits both the number of resources and selected source tokens, then injects
   labeled context at the configured message position. Resources are consumed in
   request order; semantic routing should happen upstream or in a PRA-aware
   backend.
8. **Coordinate engine lifecycle.** It prepares or reuses the backend session,
   propagates prefix handles and affinity keys, and invalidates stale engine
   state after model, system-prefix, resource-version, or explicit session
   changes.
9. **Forward generation and preserve streaming.** The transformed request is
   sent through the selected engine adapter. Streaming responses remain
   OpenAI-compatible server-sent events.
10. **Commit and report the turn.** Successful output is added to session history.
   The response trace records the effective transport, selected resource IDs,
   downgrades, bytes sent, delta operations, prefix reuse, engine-session reuse,
   and whether Native Memory was used.

### Gateway and engine responsibilities

| Gateway owns | Engine owns |
| --- | --- |
| Protocol translation and capability negotiation | Tokenization and model execution |
| Tenant/session identity and logical resource deltas | Attention and generation |
| Explicit fallback and context placement | Native K/V encoding, residency, and materialization |
| Prefix/session lifecycle coordination | Engine cache allocation and scheduling |
| Transport traces and downgrade visibility | Device memory and completion performance |

Native K/V never crosses the gateway protocol. The gateway can request or report
Native Memory, but only a capable engine adapter may create and consume it inside
the engine process. This keeps model-specific tensors out of application payloads
and prevents selected detail from being mistaken for ordinary prefix-cache state.

## Three kinds of reuse

PRA avoids sending selected context again when it is already active, lets the
inference engine reuse ordinary prefix cache where available, and can reuse
native semantic memory on qualified integrations.

| Reuse | Authority | Correctness role | Turn diagnostic |
| --- | --- | --- | --- |
| Already visible | PRA logical session ledger | Prevent duplicate context injection | `already_visible_resources`, `visible_reuse_tokens` |
| Engine prefix cached | Engine cache observation | Physical compute optimization only | `prefix_reuse_status`, `prefix_cached_tokens` |
| Native semantic memory | Qualified model/engine integration | Attach selected model-native state | `native_available_resources`, `native_attach_bytes` |

The three quantities are disjoint. A prefix hit never proves that selected
evidence is logically visible, and logical reuse never claims that K/V remained
resident. Worker, model-fingerprint, and engine-restart changes downgrade
unproven prefix observations to `UNKNOWN`.

## Operating modes

| Mode | Input accepted | Backend request | Use it when |
| --- | --- | --- | --- |
| `passthrough` | Ordinary chat | Ordinary chat | PRA mediation is disabled or handled elsewhere. |
| `selected-context` | Chat plus typed resources | Selected resources rendered as labeled text | The backend is an ordinary model endpoint. |
| `upgrade` | Compatible message metadata | Typed resource request | The backend supports typed resources but the caller uses an older message shape. |
| `typed-transport` | Chat plus typed resources and deltas | Typed resources, task/session metadata, and policy | The backend advertises PRA protocol support. |

Fallback is never silent. A request that requires an unsupported capability
fails unless text fallback is explicitly allowed and the gateway is running in
`selected-context` mode. The protocol trace then records the downgrade.

## Selected Context gateway

Use this with an ordinary engine:

```bash
pra gateway serve \
  --mode selected-context \
  --backend vllm \
  --backend-url http://127.0.0.1:8000/v1
```

The agent can send typed resources to the gateway. The gateway selects and
renders authorized resources as labeled text before calling the downstream
engine.

## Typed transport gateway

Use this only when the downstream endpoint advertises typed resources:

```bash
pra gateway serve \
  --mode typed-transport \
  --backend sglang \
  --backend-url http://127.0.0.1:30000
```

The gateway preserves full or delta resource inventories, task/session metadata,
and policy. Native K/V still remains inside the engine process.

## Sessions and resource deltas

Set `--sessions-dir PATH` to persist gateway session state with the local session
service. Within a session, the gateway tracks canonical messages, the actual
serialized backend history, compatible visible materializations, resource
versions, prefix-cache observations, and cache-affinity keys. Durable logical
snapshots reconstruct visibility after a gateway restart but intentionally do
not restore opaque engine sessions or cache handles. When the engine advertises
incremental messages and resource deltas, later
turns can send only new messages and `add`, `update`, or `remove` resource
operations.

Session state is keyed by tenant, session, and model. Inspect or explicitly close
a session with:

```text
GET    /v1/pra/sessions/{session_id}?tenant_id=TENANT&model=MODEL
DELETE /v1/pra/sessions/{session_id}?tenant_id=TENANT&model=MODEL
```

Closing a session also asks the adapter to release its corresponding engine
session. Capability changes, stale resource versions, and incompatible prefix
changes trigger invalidation and full resynchronization instead of reusing
uncertain cache state.

## Fallback placement

`--fallback-injection` controls where selected text is placed for an ordinary
backend:

| Value | Placement |
| --- | --- |
| `before_current_user` | Prepends context to the current user message; this is the default. |
| `system_suffix` | Appends context to an existing system message, or creates one. |
| `tool_context` | Inserts a tool-role context message before the current user turn. |
| `append_context_record` | Inserts a separate user-role PRA context record. |

`engine_native` is reserved for a native engine path and is rejected as a text
fallback placement.

## Capability and health endpoints

- `GET /health`
- `GET /v1/pra/capabilities`
- `POST /v1/pra/generate`
- `POST /v1/chat/completions`
- `GET /v1/pra/sessions/{session_id}`
- `DELETE /v1/pra/sessions/{session_id}`

A reachable endpoint that lacks PRA capabilities can be treated as ordinary. An
unreachable endpoint is an error. Capability loss after reconnect invalidates
cached transport state and forces a full resynchronization.

## Security boundary

The gateway rejects cross-tenant resources and credential-like PRA metadata.
Credentials remain in transport headers or provider configuration. Relevance,
cache residency, and resource sharing never grant authorization.

Operate upstreams, capability handshakes, hashed session state, and policy over
the separate, default-off [Gateway Management API](../management/gateway-rest-api.md).
See [Protocol](../protocol.md) for the inference JSON contract.
