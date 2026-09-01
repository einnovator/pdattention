# Agent Deployment

The agent documentation now has a dedicated section. Start with the [agent
integration matrix](../agents/index.md), or open the [PRA Agent
guide](../agents/pra-agent.md) for the first-party CLI, TUI, profiles, sessions,
tools, and skills.

The PRA agent keeps the conversation spine ordinary while storing documents,
tools, skills, results, and task state as typed records. This prevents a long
session from repeatedly serializing every prior artifact into each prompt.

## Start an agent

```bash
pra agent chat Qwen/Qwen3-0.6B -w . -t "Inspect this repository"
pra agent run -p work "Continue the current task."
pra agent inspect -p work
```

Profiles keep model, engine, endpoint, transport, workspace, tool authorization,
and session policy together. The local session service resolves sessions by
user and session ID and persists typed state without retaining live model K/V
indefinitely.

## Tools and skills

Tools are callable capabilities with compact selection descriptions and exact
schemas. Skills can be supplied as objects or discovered from OpenAI- or
Anthropic-style skill directories. Their complete instructions are lazy by
default: the model sees a bounded palette first and activates one exact
definition only after selection.

Tool visibility and tool execution permission remain separate. The host must
authorize side effects even when a tool is relevant.

## Result records

Tool and API results are stored as exact, typed backing records. Type-aware
compact views and address indexes make large tables, logs, graphs, and terminal
output searchable. The model can request a bounded region or cursor page rather
than re-ingesting the entire result.

## Production checklist

- Bind every record to tenant, user, session, and task scope.
- Keep provider-required tool messages on the ordinary conversation spine.
- Require explicit authorization for writes and external side effects.
- Set ingestion and native-index budgets by record type.
- Release ephemeral native state on session close.
- Preserve exact backing and provenance across compaction.

Use [Typed PRA Transport](../protocol.md) when the agent and gateway run in
different processes.
