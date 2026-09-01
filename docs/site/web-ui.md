# PRA Agent Web UI

The PRA Agent Web UI is a local browser workspace over the same agent profiles,
session service, typed records, task state, tools, and runtime used by
`pra agent chat`. It does not introduce a second agent implementation.

!!! warning "Current deployment boundary"
    The Web UI is an experimental local operator surface. It has no production
    authentication layer. Keep it on loopback unless an authenticated reverse
    proxy and appropriate network controls are in place.

## Install

Install PRA with the optional web dependencies:

```bash
python -m pip install -e ".[web]"
pra doctor
pra agent inspect
```

The extra installs FastAPI, Uvicorn, and `psutil`. `psutil` lets detached mode
verify process identity before stopping a saved PID.

## Start the UI

Start in the foreground and open the browser manually:

```bash
pra agent start --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). To launch the browser
automatically, add `--open`:

```bash
pra agent start --open
```

Run as a managed background process:

```bash
pra agent start --detach --open --json
pra agent stop
```

Detached state is stored under `PRA_HOME`, which defaults to the current
workspace's `.pra/` directory:

```text
.pra/agent-web.json
.pra/agent-web.pid
.pra/agent-web.log
```

## Select a model and profile

The Web UI resolves the same profile layers as the TUI: package defaults, user
configuration, project configuration, command configuration, then explicit
overrides.

```bash
pra agent start --profile work --config pra.yaml --pra BALANCED --open
```

| Option | Meaning |
| --- | --- |
| `--profile`, `-p` | Named agent profile containing model, engine, tools, sessions, and context policy. |
| `--config`, `-c` | Explicit YAML profile file. |
| `--pra`, `-P` | PRA bundle, profile, or runtime-configuration override. |
| `--host`, `-h` | Bind address. The safe default is `127.0.0.1`. |
| `--port` | HTTP/WebSocket port; default `8765`. |
| `--detach`, `-d` | Start a background process and persist verified lifecycle state. |
| `--open`, `-o` | Open the selected URL in the default browser. |
| `--verbose`, `-v` | Include profile-resolution detail in startup output. |
| `--json` / `--yaml` | Emit script-stable detached server state. |

Use [Model Support](models.md) to determine whether the selected model needs a
structural adapter, then inspect the complete profile before launch:

```bash
pra inspect Qwen/Qwen3-1.7B --engine hf
pra agent inspect --profile work --config pra.yaml --json
```

## Workspace layout

The browser provides three dockable areas:

- **Conversations** lists active presentation sessions.
- **Chat** streams user and assistant events over a session WebSocket.
- **Inspect** switches between task state, typed context records, and runtime
  identity.

Creating a conversation resolves the selected profile and opens a durable PRA
session. Sending a message uses the same agent turn loop as the CLI. Tool calls
that require approval produce a typed approval event; approval applies only to
that request and does not create a lasting authorization rule.

## HTTP and WebSocket surface

The server exposes operational and application endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness probe. |
| `GET /api/profiles` | Redacted profile discovery. |
| `GET/POST /api/sessions` | List or create active sessions. |
| `GET/DELETE /api/sessions/{id}` | Inspect or close one session. |
| `POST /api/sessions/{id}/messages` | Run one agent turn. |
| `POST /api/sessions/{id}/approvals` | Resolve a pending tool approval. |
| `POST /api/sessions/{id}/cancel` | Request cancellation. |
| `WS /ws/sessions/{id}` | Ordered session events with reconnect cursor support. |
| `GET /api/docs` | FastAPI's local API documentation. |

The WebSocket is a presentation channel. Durable conversation, task, and typed
record state remains owned by the configured session service.

## Security and operations

- Bind to loopback by default; the application currently has no user login.
- Keep credentials in referenced files or environment variables, not profile
  output or browser state.
- Treat tool visibility and tool execution authorization as separate controls.
- Keep tenant, user, session, and task identity in record and cache keys.
- Close sessions to release ephemeral native state.
- Put TLS, authentication, rate limits, request-size limits, and audit policy in
  a production gateway or reverse proxy.

## Troubleshooting

Check readiness and logs:

```bash
curl http://127.0.0.1:8765/health
pra agent inspect --profile work
```

For detached mode, inspect `.pra/agent-web.log`. `pra agent stop` refuses to
terminate a reused PID that does not belong to the PRA Web UI process. If the
process has already exited, stale lifecycle files are cleaned safely.

If the UI starts but generation fails, validate the model/engine pair with
`pra inspect`, then run the equivalent noninteractive request with
`pra agent run`. This separates browser transport from profile or model issues.
