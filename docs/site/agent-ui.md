# Agent profiles and interactive surfaces

Agent configuration is separate from `pra.yaml`. PRA runtime configuration
describes model and context execution; an agent profile describes endpoint,
workspace, sessions, tool authorization, skills, MCP servers, task policy, and
generation budgets.

```yaml
version: 1
default_profile: work
profiles:
  work:
    model: Qwen/Qwen3-4B-Instruct
    runtime:
      mode: gateway
      engine: vllm
      endpoint: http://mac:8000
    pra:
      profile: BALANCED
    workspace: ~/work/project
    sessions:
      path: ~/.local/share/pra/sessions
    tools:
      approval: ask
      allow_writes: true
      allow_destructive: false
      candidates: 12
      max_rounds: 6
    skills:
      directories:
        - ~/.agents/skills
        - ~/work/project/.pra/skills
    mcp:
      file: ~/.config/pra/mcp.json
    tasks:
      scope_policy: task_adaptive
    context:
      records: 16
      transport: auto
      allow_text_fallback: true
      require: [logical_refs, typed_records]
    generation:
      max_new_tokens: 2048
```

User profiles live at `~/.config/pra/agents.yaml`; project profiles live at
`./.pra/agents.yaml`. An explicit `-c/--config` has higher precedence.
`-p/--profile` selects an agent profile. `-P/--pra` independently overrides the
PRA runtime profile, bundle, or configuration.

```bash
pra agent chat -p work
pra agent run -p work "Review the failing tests."
pra agent inspect -p work --yaml
pra agent chat -p work --context-transport auto
pra agent run -p work --context-transport text "Run the compatibility baseline."
```

The startup summary and `/runtime` command show the resolved context transport,
capability source, gateway mode, engine integration level, and native-K/V
support. AUTO negotiates once at launch/session reconnect. A PRA-capable gateway
receives typed records even when it later performs a G10 text downgrade for its
ordinary downstream engine. See [Agent/Gateway Protocol](protocol.md).

Credential files remain server-side references. Their values are not placed in
session state, profile output, benchmark registries, or model cards.

## Experimental web UI

Install the optional web surface and start it on loopback:

```bash
python -m pip install -e ".[web]"
pra agent start -p work -o
pra agent start -p work -d
pra agent stop
```

The FastAPI/WebSocket server reuses `PRAAgent`, `AgentLauncher`,
`SessionService`, task services, and tool authorization. It supports multiple
durable conversations and sequence-based event replay after reconnect. The
jQuery, Bootstrap, Dockview, and Lucide frontend is a presentation layer, not a
second agent implementation. Dangerous tool calls emit a typed one-shot
approval request and remain denied unless the backend receives that approval.

`POST /api/sessions` may override only the safe transport controls for that
session. For example, the following request requires the typed PRA protocol and
disallows silent text fallback while leaving endpoint credentials and claimed
capabilities under the selected server-side profile:

```json
{
  "profile": "work",
  "session_id": "native-required",
  "context_transport": "pra",
  "allow_text_fallback": false
}
```

The default host is `127.0.0.1`. Binding to a public interface emits a warning;
the experimental UI does not claim production authentication or hardening.
