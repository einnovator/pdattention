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

## Runtime commands

```bash
pra runtime inspect Qwen/Qwen3-1.7B -e hf --storage balanced
pra runtime doctor -e hf
pra runtime serve Qwen/Qwen3-1.7B -e hf --storage balanced
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

## Operational requirements

- Pin model, tokenizer, adapter, and source versions.
- Resolve authorization before selection and again before physical attach.
- Keep session and tenant identity in cache keys.
- Cleanup on success, failure, cancellation, and disconnect.
- Record missing telemetry as `NOT_MEASURED`.
- Use the same frozen selection when comparing text and native execution.

See [Storage](../storage.md) and the [API reference](../api/agents.md).
