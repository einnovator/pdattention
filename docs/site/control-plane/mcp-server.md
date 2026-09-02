# Control Plane MCP Server

The MCP server calls `ControlManager` directly. It does not loop through REST.
It supports local stdio and authenticated streamable HTTP through the official
Python MCP SDK.

## Security defaults

- MCP is disabled by default.
- Both transports are disabled until explicitly configured.
- The default tool set is read-only plus plan inspection.
- `pra_apply` and `pra_experiment` are denied by default.
- Remote non-loopback HTTP requires bearer/OIDC/client-credential or mTLS
  profile configuration; engine credentials are never returned.

## Start stdio

```bash
pra control mcp --config control-plane.yaml --transport stdio
```

## Start streamable HTTP

```bash
export PRA_CONTROL_MCP_TOKEN='replace-with-secret-provider-value'
pra control mcp --config control-plane.yaml --transport http
```

The configured endpoint is `/mcp` by default. Use TLS at a reverse proxy when
exposing it beyond localhost.

## Tools

| Tool | Purpose | Default |
| --- | --- | --- |
| `pra_fleet` | List/filter managed instances | Enabled |
| `pra_engine` / `pra_gateway` | Inspect one instance | Enabled |
| `pra_catalog` | Read models, bundles, profiles, compatibility | Enabled |
| `pra_qualification` | Read support status and evidence | Enabled |
| `pra_deployment` | Read desired state | Enabled |
| `pra_metrics` | Read semantic metrics and observability links | Enabled |
| `pra_context` | Assemble deterministic task context | Enabled |
| `pra_plan` | Inspect an operational change | Enabled |
| `pra_apply` | Apply a confirmed plan | Disabled |
| `pra_experiment` | Read/submit research runs | Disabled |

## Resources

Stable records are exposed as `pra://fleet`, engine/model resources, catalog
models and bundles, qualifications, and deployments. Configuration filters
resource templates before MCP discovery.

`pra_context` accepts a task and optional repository. It deterministically
combines matching fleet state, bundles, qualifications, deployments, evidence,
and current limitations into one inspectable payload.
