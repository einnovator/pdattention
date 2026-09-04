# Runtime / SDK Deployment

Use the Python SDK when the application needs direct control over typed records,
sessions, tasks, routing, materialization, storage, or traces.

## Runtime construction

```python
from pra_hf import ContextPolicy, PRARuntime

runtime = PRARuntime.from_pretrained(
    "Qwen/Qwen3-1.7B",
    context_policy=ContextPolicy(
        max_native_index_tokens=4096,
        max_native_index_bytes=64 * 1024,
    ),
)
session = runtime.open_session(
    session_id="case-42",
    user_id="user-1",
    tenant_id="tenant-1",
)
```

The runtime keeps selection and physical execution separate. It can freeze
selected identities, plan model-specific materialization, and inspect every
lifecycle decision without changing task semantics.

## Session-aware realization

The gateway and embedded runtime share one realization planner. For each
selected resource it checks resource ID, version, interval, rendering profile,
and rendering digest against the model-visible session ledger. A compatible
active occurrence is preserved in place; a dropped, superseded, removed, or
task-closed occurrence does not satisfy the new turn.

Turn traces expose `requested_mode`, `resolved_mode`, selected and already-visible
resource IDs, newly materialized tokens, native attachment bytes, and physical
prefix-cache observations. Logical visibility, engine prefix reuse, and Native
Memory reuse remain separate counters so reports cannot count the same token as
two kinds of savings.

The ledger is reconciled from the actual serialized context after compaction or
history rewrite. Its durable snapshot can be restored after a process restart;
engine handles and inferred physical residency are deliberately discarded.

## Runtime commands

```bash
pra runtime inspect Qwen/Qwen3-1.7B -e hf --storage balanced
pra runtime doctor -e hf
pra runtime serve Qwen/Qwen3-1.7B -e hf -m auto --explain --storage balanced
pra runtime benchmark Qwen/Qwen3-1.7B -e hf -o .pra/bench
```

Named storage profiles are `memory`, `balanced`, `persistent`, and `minimal`.
Detailed YAML can override tier quotas, record-type retention, task-aware
priority, compression, and quantization.

## Profile behavior

`BALANCED` means every eligible native consumer layer unless a held-out,
model-specific calibration has qualified a smaller set. A candidate discovered
on a smoke cohort remains `CALIBRATION_PENDING`; the SDK does not promote it
from its name or memory reduction alone.

For RAG and multi-record context, the qualified native mode is narrower than
the record API itself. A frozen selection serialized and encoded as one
unchanged contiguous block can be retained and reused with measured semantic
parity. Independently encoded records may still be addressed, routed, and
inspected, but recomposing them into a changed order or selection is
`CALIBRATION_PENDING`. Partial materialization and the current
query-conditioned repair policy have the same status because five-seed and
held-out quality gates did not pass. Request an explicit research profile for
those modes; the runtime must not silently substitute them for contiguous
qualified execution.

## Operational requirements

- Pin model, tokenizer, adapter, and source versions.
- Resolve authorization before selection and again before physical attach.
- Keep session and tenant identity in cache keys.
- Cleanup on success, failure, cancellation, and disconnect.
- Record missing telemetry as `NOT_MEASURED`.
- Use the same frozen selection when comparing text and native execution.

See [Storage](../storage.md) and the [API reference](../api/agents.md).
