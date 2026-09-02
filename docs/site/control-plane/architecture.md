# Control Plane Architecture

The Control Plane has one application layer and several presentation adapters.

```text
browser -> REST/WebSocket --+
coding agent -> MCP --------+-> ControlManager -> Registry backend
operator -> embedded CLI ---+                  -> engine/gateway backend
built-in PRA Agent ---------+
```

The dependency direction is presentation to manager to backend. Managers do
not import FastAPI or MCP classes. Backends hide whether state and actions use
direct HTTP today or a future Registry read model, engine agent, event bus, or
broker.

## Ownership

| Layer | Owns | Does not own |
| --- | --- | --- |
| Presentation | Authentication mapping, protocol validation, rendering | Action policy, drift semantics, audit rules |
| Manager | Authorization, validation, drift, plan/apply, idempotency, audit | HTTP status codes, MCP framing, UI state |
| Backend | Registry and engine transport, physical request execution | Caller policy or presentation behavior |

The browser remains REST/WebSocket based. MCP is intended for coding agents and
automation, not as an internal transport for the web application.

## Request flow

`CallerContext` carries subject, roles, resolved permissions, tenant, request
ID, trace ID, transport, and non-secret metadata. Every manager method checks
its required permission. Mutations pass through a central audit manager and
preserve W3C trace metadata when supplied.

Operational changes use `plan -> inspect -> confirm -> apply`. Plans are
durable in the Control Plane database, expire after the configured TTL, and
support idempotency keys. This makes retries consistent across REST, MCP, and
the embedded CLI.
