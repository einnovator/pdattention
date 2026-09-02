# Control Manager Layer

`ControlManager` is the canonical facade:

```python
manager.fleet
manager.registry
manager.deployments
manager.actions
manager.qualifications
manager.observability
manager.audit
manager.context
manager.experiments
```

Managers return transport-neutral Pydantic domain models or JSON-compatible
domain records. They raise domain errors such as `NotFound`, `Forbidden`,
`Offline`, `NotSupported`, and `ApprovalRequired`. REST maps these to status
codes; MCP returns structured tool errors; the CLI renders them for operators.

## Backend boundary

`ObservedStateBackend`, `ActionBackend`, and `RegistryBackend` protocols isolate
manager semantics from transport. The current direct backend is `FleetService`,
which calls engine Management APIs and the Registry. Presentation code never
receives an engine client or management credential.

## Plan and apply

```python
plan = await manager.actions.plan(
    caller,
    "prefetch",
    "mlx-01",
    {"resource_id": "document-42"},
    idempotency_key="launch-42",
)

result = await manager.actions.apply(
    caller,
    plan.plan_id,
    reason="prepare the shared document",
)
```

The manager validates action support, capability policy, risk, permission,
confirmation, expiry, and idempotency before dispatch. High-impact operations
such as eviction, demotion, unload, and reconciliation require confirmation and
their stronger permission.
