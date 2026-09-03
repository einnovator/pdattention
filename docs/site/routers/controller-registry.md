# Controller and Registry

The Registry persists six common resources. Every router adapter consumes the same graph.

| Resource | Meaning |
|---|---|
| `RouterInstance` | Router kind, management/inference addresses, region, health, features, and revisions |
| `Route` | Stable public model or explicit LLM/MCP/A2A route |
| `ModelPool` | Model identity plus eligibility selectors |
| `BackendEndpoint` | Engine deployment advertised to one or more pools |
| `RoutingPolicy` | Deterministic constraints, preferences, weights, and fallback |
| `RouteBinding` | Assignment of a route to a router instance |

## Eligibility

The desired-state compiler checks pool membership, immutable model revision, health, maintenance state, engine, region, cluster, bundle, profile, required PRA modes, qualification tier, approval, and labels. Rejected endpoints remain visible with an exclusion reason.

The output contains deployment candidates, not a per-request replica decision.

## Reconciliation

```text
Registry change
  -> compile desired state
  -> read observed router state
  -> preview structural diff
  -> apply
  -> read back and verify
  -> record observed revision or error
```

Commands:

```bash
pra registry routers
pra registry routes
pra registry model-pools
pra registry backend-endpoints
pra registry router-desired ROUTER_ID

pra router preview ROUTER_ID
pra router reconcile ROUTER_ID --confirm
pra router controller --interval 10
```

Normal reconciliation is idempotent. A failed update leaves the router's previous configuration active and records `DEGRADED` plus the error in Registry observed state.

## Secret boundary

Registry records only `credential_reference`, normally an environment-variable or secret-manager name. It never returns the referenced token through Control Plane, MCP, route configuration, or logs.
