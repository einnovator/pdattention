# Enterprise PRA Control Plane

The eInnovator PRA Control Plane is the commercial fleet and governance layer
for PRA deployments. It coordinates open engine Management APIs and the open
PRA Registry without hiding either interface behind a proprietary protocol.

PRA engines and gateways self-register into the open Registry. The Control
Plane discovers that fleet from Registry state, so normal deployments do not
require a manually maintained endpoint list. Static and manual targets remain
available for development and disaster recovery.

```text
                 eInnovator PRA Control Plane
                SSO | RBAC | Audit | Fleet | Agent
                              |
                      PRA Registry (open)
                              |
              Gateway API + Engine APIs (open)
             /                |                \
         vLLM API          MLX API       OpenVINO API
```

| Component | Boundary | Responsibility |
| --- | --- | --- |
| Gateway Management API | Open source | Upstream health, capability negotiation, session transport, policy, and fallback state |
| Engine Management API | Open source | State and safe control for one engine instance |
| PRA Registry | Open source | Models, bundles, qualifications, policy, and desired state |
| eInnovator Control Plane | Commercial, early access | Fleet aggregation, SSO, RBAC, audit, approvals, drift UX, and governed agent assistance |

The first release is deliberately a coordinator, not an orchestrator. It can
apply safe mutable engine configuration and explain restart-required drift. It
does not replace Kubernetes, Nomad, systemd, Grafana, or Tempo.

The Control Plane aggregates gateways and engines through their public APIs.
The open Gateway API remains independently usable by the PRA CLI, Registry,
community automation, and CI; commercial enrollment is not required.

All presentation surfaces now share one semantic application layer:

```text
REST | MCP | built-in agent | embedded CLI
                    |
              ControlManager
                    |
       Registry and engine backends
```

Authorization, action policy, drift, idempotency, and audit are manager
responsibilities. REST and MCP translate protocols; neither is a second
business implementation. Start with the [architecture](architecture.md) and
[manager layer](manager-layer.md) when integrating automation.

## Current capability status

| Area | Status |
| --- | --- |
| Static, manual, and Registry fleet discovery | Early access |
| Viewer/Operator/Approver/Administrator RBAC | Early access |
| GitHub, Google, generic OIDC, local auth | Early access |
| SAML 2.0 | Optional dependency; enterprise configuration required |
| Registry catalog and approval proxy | Early access |
| Desired/observed drift | Early access |
| Audited safe engine actions | Early access |
| Fixed navigation and multi-tab Fleet workspace | Early access |
| Resumable, model-selectable PRA Agent chat | Early access, read-only assistant tools |
| Manager-backed REST and MCP | Early access; MCP is disabled and read-only by default |
| Autonomous optimization | Not shipped; recommendations require a person |

Continue with [Installation](installation.md) or review the [security and RBAC
model](rbac.md).
