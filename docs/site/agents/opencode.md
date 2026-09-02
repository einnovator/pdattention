# OpenCode

OpenCode supports custom OpenAI-compatible providers and can use the PRA
Gateway without replacing its tool loop. Version 1.18.26 completed the Stage A
fixture through Ollama/Qwen3-14B after the gateway's Chat Completions stream was
qualified for reasoning, tool-call deltas, usage, and finish reasons.

## Through the PRA gateway

Create an OpenCode configuration containing a gateway provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "pra-gateway": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "PRA Gateway",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3:14b": {
          "name": "Qwen3 14B",
          "limit": {"context": 32768, "output": 4096}
        }
      }
    }
  }
}
```

Run OpenCode in an isolated workspace with external plugins disabled:

```bash
OPENCODE_CONFIG=/path/to/opencode.json \
opencode run --pure --auto --format json \
  --model pra-gateway/qwen3:14b \
  'Inspect the repository and run its focused tests.'
```

`--auto` must be used only inside the benchmark sandbox or another workspace
where unattended tool execution is intended.

## Qualified behavior

The gateway Stage A cohort completed 4/4 exact-output tasks. OpenCode emitted
structured `tool_use` and per-step token events, which the PRA benchmark runner
normalizes. The result proves endpoint and tool-loop compatibility, not coding
quality or PRA context savings.

The first official Terminal-Bench 2.1 smoke then completed 5/5 harness runs
without exceptions but scored 0/5 task success (7/12 verifier tests passed).
That result disqualifies the general Qwen3-14B pairing from profile sweeps; it
does not disqualify OpenCode or measure a PRA effect. A code-specialized model
is the next quality gate.

## Selected Context and Native Memory

An ordinary OpenCode provider request does not include typed PRA records. It
can therefore use the gateway as a pass-through endpoint, but a plugin or
request hook is required to expose durable repository/tool-result records to
Selected Context. Native Memory additionally requires a qualified native
engine behind the gateway. Never label a pass-through run as Native Memory.

See [Agent benchmarks](benchmarks.md) for the frozen campaign design.

## Direct PRA engine

OpenCode can point the same provider configuration directly at an engine that
implements Chat Completions. This removes gateway mediation, but typed records
and Native Memory remain unavailable unless that engine and the OpenCode
request integration both negotiate those capabilities explicitly.
