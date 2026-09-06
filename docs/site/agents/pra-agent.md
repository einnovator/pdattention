# PRA Agent

PRA Agent is the first-party agent surface. It owns persistent sessions, tasks,
typed context records, tool authorization, lazy tool and skill disclosure, and
the handoff to an embedded runtime, gateway, or direct remote engine.

PRA Agent is also a benchmark subject. Its official task cohort is run against
the same PRA-capable engine in two placements: direct engine access and G11
gateway mediation. The selected records, model, profile, task sandbox, and
grader stay fixed. This pair both compares PRA Agent with endpoint-configurable
external agents and exposes first-party transport/session regressions when the
two PRA Agent outcomes disagree. It is intentionally scheduled after the
external harness and engine baselines are admitted.

The current SDK also exposes `agent.mcp`, `agent.control_plane`, `agent.targets`,
`agent.attachments`, `agent.history`, and `agent.sessions`. MCP controls remote
tools and resources; direct Control Plane REST controls deterministic fleet and
model discovery. Neither transport is required for local use.

See [Agent configuration](configuration.md), [MCP and Control Plane](mcp-control-plane.md),
and the [interactive TUI](tui.md) for the distributed client workflow.

## Start locally

The shortest path embeds the Hugging Face runtime in the agent process:

```bash
pra agent chat \
  --model Qwen/Qwen3-0.6B \
  --engine hf \
  --workspace . \
  --task "Inspect this repository"
```

For a noninteractive turn:

```bash
pra agent run --profile work "Continue the current task."
pra agent inspect --profile work
```

## Through the PRA gateway

Start the engine and a gateway, then point PRA Agent at the gateway:

```bash
pra serve Qwen/Qwen3-0.6B --engine hf --mode selected-context --port 8000
pra gateway serve \
  --mode selected-context \
  --backend openai \
  --backend-url http://127.0.0.1:8000/v1 \
  --sessions-dir .pra/gateway-sessions

pra agent chat \
  --model Qwen/Qwen3-0.6B \
  --engine hf \
  --endpoint http://127.0.0.1:8080 \
  --context-transport auto \
  --workspace .
```

`auto` reads the gateway capability endpoint. It preserves typed records when
supported and uses text only when the profile permits explicit fallback. Use
`--context-transport pra --no-text-fallback` when the workload requires typed
transport and must fail rather than downgrade.

## Direct PRA engine

For an embedded direct runtime, omit `--endpoint`. For a remote PRA-aware
runtime, point the agent at that runtime instead of the gateway:

```bash
pra serve Qwen/Qwen3-0.6B --engine hf --mode auto --port 8000

pra agent chat \
  --model Qwen/Qwen3-0.6B \
  --engine hf \
  --endpoint http://127.0.0.1:8000 \
  --context-transport pra \
  --no-text-fallback
```

The direct endpoint must advertise typed records and every capability required
by the selected profile. Native K/V remains in the engine process.

## Persistent profile

Place this in `.pra/agents.yaml`:

```yaml
version: 1
default_profile: work
profiles:
  work:
    model:
      id: Qwen/Qwen3-0.6B
    runtime:
      mode: gateway
      engine: hf
      endpoint: http://127.0.0.1:8080
    workspace: .
    sessions:
      path: .pra/sessions
      resume_last: true
    context:
      transport: auto
      allow_text_fallback: true
      records: 12
    tools:
      approval: ask
      allow_writes: false
      allow_destructive: false
    skills:
      directories:
        - .agents/skills
```

Run `pra agent inspect --profile work` before the first session to inspect the
resolved model, endpoint, profile, tool policy, and context transport.

## What remains local

The agent owns tool effects and authorization. The gateway owns protocol and
session mediation. The engine owns tokenization, attention, Native Memory, and
generation. A selected or cached record is never authorization to execute a
tool.
