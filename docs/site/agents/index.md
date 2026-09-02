# Agents

PRA can sit inside an agent, between an agent and its model endpoint, or inside
the model engine. These placements are not equivalent. An OpenAI-compatible
base URL is sufficient for ordinary chat transport, but typed records require
the agent to send the PRA extension envelope or use a family-specific bridge.

## Support matrix

| Agent | PRA gateway | Direct PRA engine | Typed-record integration |
| --- | --- | --- | --- |
| [PRA Agent](pra-agent.md) | First-class | First-class, embedded or remote | Built in |
| [DeepSeek Harness](deepseek-harness.md) | Tested SDK bridge | Tested logical request builder | Python bridge; host plugin glue required |
| [OpenCode](opencode.md) | Tested OpenAI-compatible gateway path | OpenAI-compatible endpoint | No packaged typed bridge |
| [Pi Coding Agent](pi-coding-agent.md) | Tested SDK bridge | Tested logical request builder | Python bridge; Pi extension glue required |
| [OpenHands](openhands.md) | OpenAI-compatible endpoint | OpenAI-compatible endpoint | No packaged typed bridge |
| [Aider](aider.md) | OpenAI-compatible endpoint | OpenAI-compatible endpoint | No packaged typed bridge |
| [Codex CLI](codex-cli.md) | Tested Responses translation | Requires a Responses-compatible engine | Typed request integration pending |
| [Claude Code](claude-code.md) | Current gateway is not Anthropic-compatible | Requires an Anthropic-compatible engine | No packaged typed bridge |

## Pick an integration

Use **PRA Agent** when you want the complete product path: persistent sessions,
typed records, task state, lazy tools and skills, selection, and explicit
transport negotiation.

Use the **DeepSeek Harness** or **Pi** bridge when the external agent must remain
the owner of its tool loop. Their adapters turn durable tool-result events into
stable, deduplicated PRA resources. They transport logical records, never raw
model K/V.

Use an **OpenAI-compatible endpoint integration** for OpenHands or Aider when
you first need model connectivity. That route does not automatically create PRA
records from agent history. Add a typed event bridge before claiming PRA context
selection or native reuse.

Codex CLI uses the gateway's tested Responses translation. Claude Code remains
separate because the reference gateway does not implement Anthropic Messages.
An endpoint qualification still does not imply that typed PRA records or Native
Memory were used; check the negotiated request trace.

## Two deployment paths

### Through the PRA gateway

Run an engine on one port and the gateway on another:

```bash
pra serve MODEL --engine hf --mode selected-context --port 8000
pra gateway serve \
  --mode selected-context \
  --backend openai \
  --backend-url http://127.0.0.1:8000/v1 \
  --sessions-dir .pra/gateway-sessions
```

The agent endpoint is `http://127.0.0.1:8080/v1`. A standard chat request works,
but Selected Context is meaningful only when the request includes typed PRA
resources. See the individual agent page for whether that envelope is available.

### Direct PRA engine

Point a capable client directly at the runtime endpoint:

```bash
pra serve MODEL --engine hf --mode auto --profile recommended --port 8000
```

The direct path removes gateway mediation. The client must negotiate the engine
capabilities and send typed records itself to preserve PRA semantics. Never
assume that an ordinary chat request uses Native Memory merely because the
engine supports it.

## Verify the boundary

For gateway or direct typed transport, inspect capabilities before sending a
workload:

```bash
curl http://127.0.0.1:8080/v1/pra/capabilities
```

Check `effective_capabilities`, not only the engine name. If `typed_records` is
false, use visible Selected Context or install an agent bridge.
