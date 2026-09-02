# Control Plane Operation Catalog

This page is generated from `pra_control.operations`. The same catalog drives manager authorization metadata, REST exposure, MCP discovery, agent capabilities, and audit fields.

## Operations

| Operation | Permission | Side effect | Risk |
|---|---|---|---|
| `fleet.list` | `fleet:read` | none | read |
| `engine.inspect` | `engine:read` | none | read |
| `engine.register` | `engine:configure` | write | write |
| `engine.remove` | `engine:high-impact` | write | high |
| `engine.action` | `engine:action` | write | write |
| `engine.config.patch` | `engine:configure` | write | write |
| `registry.list` | `registry:read` | none | read |
| `registry.write` | `registry:write` | write | write |
| `qualification.read` | `qualification:read` | none | read |
| `qualification.approve` | `qualification:approve` | write | high |
| `deployment.read` | `deployment:read` | none | read |
| `deployment.write` | `deployment:write` | write | write |
| `deployment.apply` | `deployment:apply` | write | high |
| `action.plan` | `engine:read` | none | read |
| `action.apply` | `engine:action` | write | write |
| `observability.read` | `observability:read` | none | read |
| `audit.read` | `audit:read` | none | read |
| `context.read` | `fleet:read` | none | read |
| `experiment.read` | `experiment:read` | none | read |
| `experiment.run` | `experiment:run` | write | high |

## MCP tools

| Tool | Operations | Default |
|---|---|---|
| `pra_fleet` | `fleet.list` | read-only |
| `pra_engine` | `engine.inspect` | read-only |
| `pra_gateway` | `engine.inspect` | read-only |
| `pra_catalog` | `registry.list` | read-only |
| `pra_qualification` | `qualification.read` | read-only |
| `pra_deployment` | `deployment.read` | read-only |
| `pra_metrics` | `observability.read` | read-only |
| `pra_context` | `context.read` | read-only |
| `pra_plan` | `action.plan` | read-only |
| `pra_apply` | `action.apply` | disabled |
| `pra_experiment` | `experiment.read`, `experiment.run` | disabled |

Deny patterns override allow patterns. Disabling discovery never grants or removes a caller permission; manager authorization is always enforced.
