# MCP And Control Plane

PRA Agent deliberately keeps two remote paths.

| Path | Responsibility |
| --- | --- |
| MCP client | Remote tools and addressable resources |
| Control Plane REST | Fleet inventory, qualifications, recommendations, and inference targets |

## MCP lifecycle

`MCPClientManager` supports multiple HTTP and stdio servers. Each server moves
through `DISCONNECTED`, `CONNECTING`, `CONNECTED`, `DEGRADED`, or `FAILED`.
An optional server can fail without blocking startup; a required server fails
startup. Every transport is entered, used, and closed by one supervisor task so
stdio/AnyIO cancellation scopes are cleaned up correctly.

Tool runtime names are globally unique:

```text
mcp:<server>:<tool>
```

MCP JSON Schema becomes a PRA `ToolRecord`. Source server, transport, original
tool name, trust, and read/write/destructive annotations remain attached. Tools
without trustworthy read-only annotations are conservatively mutating. MCP
resources preserve the original URI and source server in typed metadata.

```bash
pra agent mcp add docs http://127.0.0.1:9400/mcp \
  --config pra-agent.yaml --save
pra agent mcp list --config pra-agent.yaml --connect
pra agent mcp remove docs --config pra-agent.yaml --save
```

Inside the TUI, use `/mcp status`, `/mcp tools`, `/mcp resources`, `/mcp add`,
and `/mcp remove`. Add/remove changes are in memory until `/save`.

## Control Plane targets

Static providers and discovered models resolve to one `InferenceTarget` shape.
Its stable identity is `<engine-instance>/<runtime-model-id>`. Static endpoint
and credential references override matching discovered values; Control Plane
status, capability, and qualification metadata enrich the row.

`/models` refreshes the inventory. `/model use <target>` validates an exact or
unambiguous model name, replaces the backend, closes incompatible native model
state, and resumes the same durable logical session. Native K/V is never reused
across this model-fingerprint boundary.

When the Control Plane is unavailable, cached discovery and static targets stay
available. A required Control Plane or MCP server instead makes startup fail.
