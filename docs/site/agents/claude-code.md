# Claude Code

Claude Code speaks the Anthropic Messages protocol. The current PRA gateway
speaks OpenAI-compatible Chat Completions plus the PRA extension, so changing
`ANTHROPIC_BASE_URL` to the reference gateway is not supported.

## Through the PRA gateway

**Not supported directly.** An Anthropic-to-OpenAI translation proxy may make
ordinary model calls possible, but it does not automatically preserve Claude
Code tool blocks or add typed PRA records. Do not describe that chain as PRA
integration without an end-to-end protocol and causal test.

An eventual gateway adapter must accept Anthropic Messages, preserve tool-use
and tool-result blocks, attach stable typed resources, and translate the
response stream back to Claude Code. The reference gateway has no such endpoint
today.

## Direct PRA engine

Direct use requires a PRA-aware engine that implements the Anthropic Messages
surface expected by Claude Code and a bridge that emits typed PRA resources.
The current direct runtime does not provide that complete surface.

Use PRA Agent for the supported local agent workflow:

```bash
pra agent chat --model Qwen/Qwen3-0.6B --workspace .
```

Keep Claude Code on its supported Anthropic or documented enterprise gateway
path until a typed adapter is available. Anthropic's [LLM gateway
documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)
describes its expected base URLs and authentication.

## Adapter requirements

A future adapter must preserve Claude Code's message and tool-result ordering,
authorization prompts, cancellation, and streaming. It should capture only
committed results, assign stable record IDs, and keep Native Memory inside the
engine process.
