# Codex CLI

Codex CLI and the current PRA reference gateway do not expose the same model
protocol. The gateway implements OpenAI-compatible Chat Completions plus the PRA
extension; Codex CLI model providers use the Responses protocol. A chat base-URL
substitution is therefore not a supported integration.

## Through the PRA gateway

**Not supported by the current gateway.** Do not point Codex CLI at
`http://127.0.0.1:8080/v1` and assume tool calls, reasoning items, streaming, or
typed context will survive translation.

A complete Codex integration needs a gateway Responses endpoint that preserves
the Codex item stream and adds PRA typed resources without flattening tool
state. That endpoint is not implemented in this branch.

## Direct PRA engine

A direct route is possible only when the PRA-aware engine implements the
Responses protocol expected by the selected Codex model provider. The current
reference PRA runtime exposes its chat and typed PRA surfaces, not a complete
Codex Responses backend.

For PRA-backed repository work today, run the first-party TUI:

```bash
pra agent chat --model Qwen/Qwen3-0.6B --workspace .
```

Keep Codex CLI on its supported provider independently. Do not claim that a
shared model or an OpenAI-shaped URL constitutes PRA integration.

## Adapter requirements

A future Codex adapter must preserve Responses input/output item identity,
function-call pairing, approvals, cancellation, and streaming; convert only
durable results into typed PRA records; and negotiate fallback explicitly. It
must never place raw K/V in Codex configuration or request payloads.

Consult the official [Codex configuration reference](https://developers.openai.com/codex/config-reference)
when implementing that provider rather than copying Chat Completions settings.
