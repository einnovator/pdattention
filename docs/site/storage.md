# Semantic storage lifecycle

PRA native K/V is a derived cache of an authoritative typed source record. The
runtime assigns each native object one engine-neutral service tier:

| Tier | Meaning |
|---|---|
| `HOT` | Attention-ready engine representation, request-pinnable. |
| `WARM` | Lossless native detail available for fast promotion. |
| `COLD` | Persistent compressed and optionally quantized native detail. |
| `SOURCE` | Canonical record from which native detail can be reconstructed. |

vLLM pages, SGLang objects, and MLX arrays are physical realizations of `HOT`;
they are not separate retention policies. `PRAStorageManager` owns typed-record,
task, dependency, session, and quota decisions. An engine bridge only loads,
pins, unpins, measures, and releases `HOT` payloads.

## Configure storage

Named profiles are `memory`, `balanced` (default), `persistent`, and `minimal`:

```powershell
pra runtime serve Qwen/Qwen3-1.7B -e hf --storage balanced
pra runtime inspect Qwen/Qwen3-1.7B -e hf --storage persistent --json
```

Use YAML for detailed policy:

```yaml
storage:
  profile: balanced
  hot:
    max_bytes: 8GiB
  warm:
    enabled: true
    path: ~/.cache/pra/warm
    max_bytes: 64GiB
    compression: none
    cold_grace_seconds: 15m
  cold:
    enabled: true
    path: ~/.cache/pra/cold
    max_bytes: 512GiB
    compression: gzip
    kv_quantization: none
  eviction:
    policy: weighted_lru
    record_types:
      generic_document:
        retention_class: persistent_shared
        priority: 1.0
        warm_ttl: 7d
        cold_ttl: 90d
        cold: true
      tool_response:
        retention_class: ephemeral
        priority: 0.15
        warm_ttl: 10m
        cold: false
  tasks:
    active:
      priority_multiplier: 2.0
      min_warm_retention: 2h
    completed:
      priority_multiplier: 0.5
      compaction_delay: 5m
```

```powershell
pra runtime serve MODEL -e mlx --storage-config storage.yaml
pra runtime inspect MODEL -e mlx --storage-config storage.yaml --yaml
```

Paths default below `PRA_HOME`. Compression and K/V quantization are separate:
compression changes byte transport without changing values; quantization can
change values and therefore requires a model/profile quality gate.

## Lifecycle behavior

Newly encoded detail stays `HOT` during a persistence grace period. This avoids
writing short-lived tool output merely to delete it after task closure. Open,
blocked, and waiting tasks extend WARM retention. Completion schedules delayed
compaction; an open downstream dependency prevents that compaction. Session
closure preferentially releases transient logs and tool output while preserving
authorized shared resources.

Eviction never deletes authoritative SOURCE because a native-cache quota is
full. A reconstructable DB/RAG/tool result can fall back to `SOURCE` and be
encoded again. Request pins prevent HOT release until decode cleanup. Serving
callers pass tenant and authorization scopes to both promotion and pinning; the
second atomic check is deliberate because cache residency is not authorization.

## Inspect and reproduce

`pra runtime inspect` reports the resolved profile, tier budgets and paths,
compression, quantization, deterministic eviction policy, task awareness, and
record-type priors. An active SDK runtime additionally reports per-tier object
and byte usage plus transition and I/O counters.

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_5_runtime.run_storage_lifecycle
```

The checked-in control workload runs E1--E9 over HF, vLLM, SGLang, and MLX
policy realizations with five seeds. It tests tier recovery, quantization,
record/task/dependency/session behavior, reconstruction, and persistence-write
waste. It does not substitute for engine-native end-task quality evaluation.
