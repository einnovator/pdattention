# Desired State

The Registry may own desired deployment state while each engine's Management
API reports observed state. This separation prevents the catalog from claiming
that a model is loaded merely because it was approved.

Every deployment mutation increments `desired_revision`, including profile,
mode, storage policy, observability policy, and engine selector changes.

```text
Registry desired revision 17
             |
             v
Control Plane comparison ---- Engine observed revision 16
             |
             v
          DRIFTED
```

`GET /v1/deployments/{id}/desired` returns the complete versioned intent.
`POST /v1/resolve/deployment` chooses the highest desired revision for an
environment and cluster with stable ID tie-breaking. Reconciliation is left to
the control plane or deployment automation; the Registry does not silently
mutate engines.

## One or several loaded models

Legacy `desired_model_id`, `desired_bundle_id`, `desired_profile_id`, and
`desired_mode` fields remain valid and normalize to one runtime model named
`default`. Engines that genuinely host several models use `desired_models`:

```yaml
desired_models:
  - runtime_model_id: qwen
    model_id: Qwen/Qwen3-14B
    bundle_id: EInnovator/pra-qwen3-14b
    profile_id: BALANCED
    mode: native-memory
  - runtime_model_id: gemma
    model_id: google/gemma-3-12b-it
    profile_id: QUALITY_MAX_CANDIDATE
    mode: selected-context
allow_extra_models: false
```

The first list entry is also projected into legacy singular fields for older
clients. Reconciliation matches rows by `runtime_model_id`, reports drift for
each model, and keeps one Registry `ManagedInstance` per engine process.
