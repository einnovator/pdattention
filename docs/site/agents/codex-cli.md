# Codex CLI

Codex CLI can use the PRA Gateway through its OpenAI Responses endpoint. The
gateway preserves streamed text and reasoning, function and custom tool calls,
call/output identity, usage, and terminal completion events. This route was
qualified with Codex CLI 0.147.0 and Ollama/Qwen3-14B on an M5 Mac.

## Through the PRA gateway

Start an OpenAI-compatible backend and point the gateway at it:

```bash
pra gateway serve \
  --mode passthrough \
  --backend ollama \
  --backend-url http://127.0.0.1:11434 \
  --model qwen3:14b \
  --port 8080
```

Add a Codex provider in `~/.codex/config.toml`:

```toml
[model_providers.pra_gateway]
name = "PRA Gateway"
base_url = "http://127.0.0.1:8080/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

Then run a clean, non-persistent session:

```bash
OPENAI_API_KEY=local-benchmark codex exec \
  --ephemeral --sandbox workspace-write \
  -c 'model_provider="pra_gateway"' \
  --model qwen3:14b \
  'Inspect the repository and run its focused tests.'
```

The API key is a local placeholder when the backend does not authenticate. Do
not put a real provider secret in benchmark manifests or result artifacts.

## What is qualified

The Stage A cohort completed four exact-output tool tasks through:

```text
Codex CLI -> Responses API -> PRA Gateway -> Chat Completions -> Ollama/Qwen3-14B
```

All four tasks succeeded. This is transport and tool-loop evidence only. It is
not Terminal-Bench/SWE-bench quality evidence, and pass-through mode does not
exercise Selected Context or Native Memory.

## Direct PRA engine

A direct route is valid only when the engine itself implements the Responses
contract required by Codex. Most PRA engine adapters expose Chat Completions,
so the gateway translation above is the portable route. A shared model name or
an OpenAI-shaped URL alone does not establish compatibility.

## Limitations

- Mid-tool cancellation is not yet qualified.
- Unknown local model IDs use Codex fallback metadata; pin model context and
  output limits in controlled studies.
- Typed PRA resources still require a Codex-aware request integration. The
  Stage A pass-through run does not claim typed-record transport.
- Commercial-native Codex remains a separate quality and cost control.

See the [agent benchmark protocol](benchmarks.md) for the frozen command and
the boundary between qualification and performance evidence.
