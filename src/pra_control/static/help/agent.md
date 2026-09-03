# PRA Agent

The right pane provides a persistent, RBAC-governed assistant for fleet operations. Its toolbar resumes previous sessions, starts a new session, shows command tips, or collapses the pane. It can answer questions about engine health, drift, bundles, profiles, storage, and observability using Control Plane tools.

The model selector lists runtime models discovered from the fleet. Reachable models are marked `[online]`; unavailable models remain visible but cannot be selected. The configured default comes from `control_plane.agent`. When no model is selected, or model inference fails, the manager-only assistant continues to provide deterministic fleet answers.

## Commands

| Command | Purpose |
| --- | --- |
| `/status` | Show connection and selected model state |
| `/fleet` | Summarize fleet health and drift |
| `/models` | List model targets and reachability |
| `/model use ENGINE:MODEL` | Select a reachable runtime model |
| `/sessions` | Count retained Control Plane sessions |
| `/new` | Start a new session in the Web UI |
| `/clear` | Clear visible chat messages |
| `/tips` or `/help` | Show the command guide |

## Example questions

```text
Which engines are running Qwen3-4B?
Show deployments whose observed bundle differs from Registry intent.
Which engines have active storage alerts?
Summarize qualifications for the BALANCED profile.
```

The connection indicator reports `Connecting`, `Connected`, or a bounded retry delay. Messages resume from the last acknowledged sequence after a transient disconnect.

Tool activity is shown inline. Agent access does not bypass authorization: mutations still require the normal role, reason, confirmation, and audit path.
