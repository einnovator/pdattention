# External Coding Agents

Coding agents should use the read-only stdio profile first. Enable `pra_apply`
only for a dedicated, narrowly permissioned identity after plan review and
audit retention are configured.

## Claude Code

```bash
claude mcp add pra-control -- pra control mcp --config control-plane.yaml --transport stdio
```

## Codex

Add a stdio MCP server whose command is `pra` and arguments are
`control mcp --config control-plane.yaml --transport stdio`. Keep the service
identity at `Viewer` for repository work that only needs fleet and evidence
context.

## OpenCode

Configure a local MCP command with the same executable and arguments. Use
`pra_context` at task start, then `pra_engine` or `pra_qualification` for narrow
follow-up inspection.

## OpenHands

Run streamable HTTP behind TLS when the agent is containerized or remote. Pass
the configured bearer token as an authorization header and expose only the MCP
tool subset needed by that deployment.

All four clients see the same domain facts as REST because the tools invoke the
same manager facade. The local transport was protocol-smoke-tested with the
official Python MCP client; individual third-party client releases may use
different configuration-file syntax.
