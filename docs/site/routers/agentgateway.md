# agentgateway

agentgateway is the high-performance Rust data plane for regional LLM, MCP, and A2A traffic. PRA adds fleet discovery, qualification, bundle/profile/mode constraints, and governed desired state.

The initial standalone integration emits a schema-validated YAML/JSON configuration through an atomic watched file. agentgateway reloads routing sections without restarting; the top-level static `config` section remains startup configuration. A sidecar revision file keeps PRA reconciliation metadata outside agentgateway's strict configuration schema. The same canonical compiler is structured for a future incremental xDS transport.

```yaml
router:
  id: agentgateway-eu
  kind: agentgateway
  management_url: https://pra-config-bridge.internal
  inference_url: http://agentgateway:3000
  region: eu
  metadata:
    port: 3000
```

```bash
agentgateway -f /var/lib/pra/agentgateway-dynamic.yaml
pra router reconcile agentgateway-eu --confirm
```

For direct watched-file deployment, register the absolute configuration path as a
file URL instead of the HTTPS configuration bridge shown above. Compiled state
includes gateways, stable backend names, and weighted route targets. LLM, MCP,
and A2A remain explicit Registry route kinds; adapters can map them to separate
native sections as those routes are enabled. Registry keeps richer PRA identity
metadata because agentgateway's local schema rejects arbitrary fields, and secret
values are never emitted. For native configuration details, see the [official
agentgateway configuration overview](https://agentgateway.dev/docs/standalone/latest/configuration/overview/).

Use one RouterInstance and bindings per regional router. The Registry provides the global view; the regional agentgateway remains responsible for request-time selection and failover.
