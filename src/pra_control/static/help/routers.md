# Routers and routes

Routers are external request data planes. PRA Registry stores route intent and
qualification-aware backend eligibility; the Router Controller translates that
intent into LiteLLM, agentgateway, Kubernetes GAIE, Bifrost, or PRA Reference
Router configuration.

## Inspect drift

Open **Routing > Routers**, then select a router. The detail view compares the
Registry desired revision with the router's observed revision and lists each
configuration change before it is applied.

## Reconcile

Administrators and approvers can apply a previewed revision. Reconciliation is
explicitly confirmed and audited. If apply or read-back verification fails, the
router keeps serving its last-good configuration.

The Control Plane is never part of the inference request path. Replica choice
also remains with the external router or cluster-local endpoint picker.
