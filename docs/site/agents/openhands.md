# OpenHands

OpenHands can target an OpenAI-compatible base URL, so it can use the PRA
gateway or a direct PRA runtime as a model endpoint. The current repository does
not ship an OpenHands event bridge; base-URL compatibility alone does not turn
terminal output, files, or observations into typed PRA records.

## Through the PRA gateway

Start the gateway, then configure the OpenHands CLI environment:

```powershell
$env:LLM_BASE_URL = "http://127.0.0.1:8080/v1"
$env:LLM_API_KEY = "local"
$env:LLM_MODEL = "openai/Qwen/Qwen3-0.6B"
openhands --override-with-envs
```

Use `selected-context` only after an OpenHands callback or custom LLM wrapper
adds the versioned `pra` envelope to each request. Without that bridge, use
`passthrough`; the gateway is only an OpenAI-compatible model proxy.

OpenHands documents `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL` in its
[environment reference](https://docs.openhands.dev/openhands/usage/environment-variables).

## Direct PRA engine

Point `LLM_BASE_URL` at the engine instead:

```powershell
$env:LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:LLM_API_KEY = "local"
$env:LLM_MODEL = "openai/Qwen/Qwen3-0.6B"
openhands --override-with-envs
```

This is direct model transport. It becomes direct PRA transport only when the
OpenHands LLM layer sends typed resources and verifies the engine capability
handshake. Native support in the engine does not reinterpret ordinary OpenHands
history automatically.

## Add typed records

The integration point is an OpenHands SDK conversation callback plus a custom
LLM request wrapper. Capture final, durable observations rather than streaming
fragments; keep tool authorization in OpenHands; assign stable record IDs; and
place the records in the OpenAI request's `pra.resources` field. Reuse the same
selector output when comparing gateway-rendered context with direct Native
Memory.

Until that adapter is implemented and tested, the supported OpenHands status is
**OpenAI-compatible endpoint only**.

## Benchmark qualification

Harbor 0.22.0's installed adapter was compatible with OpenHands 0.57.0, not
the audited current 1.11.0 package layout. The pinned agent reached
Ollama/Qwen3-14B through the PRA Gateway on Terminal-Bench's `query-optimize`
task, but repeated the harness continuation response, produced no `sol.sql`,
passed only the two pre-solution verifier checks, and received reward zero.
This is a valid negative compatibility result for that pinned stack. It is not
a PRA comparison or a claim about newer OpenHands releases.
