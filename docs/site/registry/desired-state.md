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
