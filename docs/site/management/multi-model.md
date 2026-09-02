# Engine Instances and Loaded Models

PRA separates the identity of an engine process from the identity of each model
resident inside it:

```text
EngineInstance
  |-- LoadedModel qwen3:14b
  |     |-- PRA bundle and profile
  |     |-- model-scoped storage
  |     `-- model-scoped sessions
  `-- LoadedModel gemma3:12b
        |-- PRA bundle and profile
        |-- model-scoped storage
        `-- model-scoped sessions
```

Single-model engines are simply the one-model case of this API. Their runtime
ID defaults to `default`, and their existing `ManagementProvider(models=[...],
storage_manager=..., session_source=...)` construction remains valid.

## Runtime model identity

`runtime_model_id` is the stable local name used inside one engine instance. It
is distinct from a global Hub model ID. For example, an Ollama runner may use
`qwen3:14b`, an OVMS deployment may use its configured name and version, and a
llama.cpp router may use a route alias.

Use `GET /v1/pra/models/{runtime_model_id}` for local identity and
`GET /v1/pra/models?model_id=Qwen/Qwen3-14B` for global lookup.

## Engine behavior

| Engine mode | Loaded models per instance | Dynamic lifecycle |
| --- | ---: | --- |
| vLLM base server, HF/reference, SGLang, MLX, TensorRT-LLM | 1 | Process replacement when required |
| Ordinary llama-server | 1 | No |
| llama.cpp router mode | 1..N | When the router implements it |
| OpenVINO Model Server | 1..N | Only through supported OVMS operations |
| Ollama | Active or runner-known models only | When attached by the adapter |
| AirLLM and FreeToken | 1 | No |

Installed or downloadable models are catalog entries, not loaded models.
`GET /models` never reports every model present on disk.

## Desired state

Legacy deployment fields normalize to one `default` desired model. Multi-model
instances use a list:

```yaml
desired_models:
  - runtime_model_id: qwen3:14b
    model_id: Qwen/Qwen3-14B
    bundle_id: EInnovator/pra-qwen3-14b
    profile_id: BALANCED
    mode: selected-context
  - runtime_model_id: gemma3:12b
    model_id: google/gemma-3-12b-it
    profile_id: QUALITY_MAX_CANDIDATE
    mode: selected-context
allow_extra_models: false
```

Drift is computed per runtime model and then aggregated. Missing desired models
report `MODEL_NOT_LOADED`; disallowed additional models report
`UNAPPROVED_MODEL_LOADED`.

## Gateway routing

One upstream can advertise several `(runtime_model_id, model_id)` mappings. The
gateway first finds eligible loaded models for the requested global model, then
selects the `(instance_id, runtime_model_id)` target. Session affinity retains
that complete pair rather than assuming one upstream URL means one model.

## Isolation

Model-specific storage and session sources live in separate
`ModelRuntimeState` objects. Native cache identity includes model fingerprint,
resource identity and version, span or selected representation, and layout.
Changing or unloading a model invalidates incompatible native state while
independently owned SOURCE records may remain shared.
