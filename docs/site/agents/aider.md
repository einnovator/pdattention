# Aider

Aider supports OpenAI-compatible model endpoints. It can therefore reach a PRA
gateway or a direct runtime, but the stock client does not send typed PRA
resources. Treat the current integration as model transport rather than a
typed-record plugin.

## Through the PRA gateway

Start the gateway and launch Aider with its OpenAI-compatible options:

```bash
aider \
  --openai-api-base http://127.0.0.1:8080/v1 \
  --openai-api-key local \
  --model openai/Qwen/Qwen3-0.6B
```

Use gateway `passthrough` for the stock client. A `selected-context` gateway
needs an additional Aider wrapper that converts repository artifacts and tool
results into the request's `pra` envelope; otherwise there are no resources to
select.

See Aider's [OpenAI-compatible API guide](https://aider.chat/docs/llms/openai-compat.html)
for its current base-URL and model naming rules.

## Direct PRA engine

Point the same option at the direct runtime:

```bash
aider \
  --openai-api-base http://127.0.0.1:8000/v1 \
  --openai-api-key local \
  --model openai/Qwen/Qwen3-0.6B
```

This uses the direct engine for ordinary chat. It does not activate typed
selection or Native Memory by itself. A future Aider bridge must preserve edit
and tool-result identity, attach typed records before the provider call, and
keep Aider's confirmation policy authoritative.

## Recommended use today

Use Aider with the gateway when endpoint centralization is useful, or directly
with a qualified engine for ordinary model access. Use [PRA Agent](pra-agent.md),
[DeepSeek Harness](deepseek-harness.md), or [Pi](pi-coding-agent.md) when typed
agent-result retention is required today.
