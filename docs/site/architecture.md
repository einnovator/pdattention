# Architecture

## Architecture at a glance

PRA sits between application state and model execution. The application keeps
the full logical record stream; the runtime builds a bounded active working set
for each operation.

```text
Agent/application
        |
typed records + tasks + sessions
        |
PRA runtime
  route
  select
  materialize
        |
Gateway / SDK / native engine
        |
Inference engine
```

The three runtime operations have independent responsibilities:

1. **Route** scores compact addresses against the current query.
2. **Select** applies identity, authorization, task scope, and budgets.
3. **Materialize** converts the frozen selection to text, typed resources, or
   qualified model-native memory.

This separation lets the same selector feed a portable Selected Context path
and a Native Memory path. It also makes comparisons meaningful: representation
changes without silently changing retrieved evidence.

## Ownership boundaries

| Component | Owns |
| --- | --- |
| Agent/application | User intent, messages, tasks, tool authorization |
| PRA runtime | Typed records, routing, selection, exact backing, lifecycle policy |
| Gateway | Capability negotiation, deltas, explicit fallback, wire traces |
| Engine adapter | Model revision, K/V geometry, positions, masks, cleanup |
| Serving scheduler | Placement, prefetch, batching, sharing, eviction |

Raw K/V remains below the gateway boundary. The wire protocol carries stable
resource identities and policy, not engine page handles or tensors.

## Record flow

A record enters with type, version, source fingerprint, tenant/session scope,
provenance, and authorization. Type-specific compactors build:

- an initial compact view safe for the active context;
- address views for lexical, semantic, entity, schema, or structural search;
- exact backing detail for later materialization.

Selection returns stable record IDs and bounded regions. Materialization
rechecks scope and authorization, retrieves exact detail, and records measured
cost. Native ingestion is size-gated; an oversized record remains searchable
and can encode only an authorized selected region lazily.

## Session and task lifecycle

Sessions are resolved by user and session identity and retain a typed record
stream across turns. Tasks form a versioned dependency graph. Task status can
adjust selection scope and storage retention without changing record ownership.

Closing an engine session releases request-local and ephemeral native state.
The logical session and authoritative backing can remain durable. Cache
presence never grants access to another tenant or session.

## Deep internals

The rest of this page explains how a native-capable model consumes a frozen
selection. These details are not required for Selected Context deployments.

### Reference identity and tables

Explicit handles and runtime-created record references resolve through a scoped
reference table to immutable source versions. Recursive expansion is bounded by
depth, reference count, token count, cycle detection, and missing-reference
policy.

### Chunks, gists, and routing

Each retained source is partitioned into routing chunks. One or more compact
gists represent each chunk or resource. A projected query scores those gists,
then selects resources and chunks under independent budgets:

```text
query [batch, model_width]
  -> resource gist scores [resources, gists]
  -> selected resource IDs
  -> chunk gist scores [chunks, gists]
  -> selected source intervals
```

The compact routing index identifies source intervals. It does not replace the
exact token detail used by the model.

### Layer-specific native memory

For each consuming decoder layer, the model encodes selected detail with that
layer's own projections and positional policy. A typical cache tensor is:

```text
K, V: [batch, kv_heads, selected_tokens, head_width]
```

Native memory must match model revision, tokenizer, layer, dtype, head geometry,
position policy, and source version. A mismatch invalidates reuse.

### Positions and long prompts

The direct prompt tail stays in ordinary causal attention. Displaced history can
be represented as a request-local implicit source and routed through the same
selection path as explicit resources. Source-relative and query-relative
positions remain distinct; adapters must preserve the host model's positional
geometry.

### Attention consumption

A consumer layer computes ordinary local attention and selected-memory
attention over the same hidden query. Implementations may concatenate K/V or
use a mathematically equivalent segmented normalization. The selected and local
segments must share one normalization if they are intended to compete directly.

### Batch isolation

Every batch row owns a separate reference namespace. Selected lengths can vary
by row; implementations bucket or pad physical tensors, mask invalid positions,
and restore row order. No routing or materialization operation may consult
another row's identities.

### Storage and serving

Native objects can move between attention-ready, warm lossless, cold persistent,
and reconstructible source states. Request pins prevent eviction during decode.
Native Serving additionally gives the scheduler ownership of prefetch, sharing,
placement, and batching while preserving tenant/session authorization.

See [Storage](storage.md) for lifecycle policy and [Protocol](protocol.md) for
the application-to-engine boundary.
