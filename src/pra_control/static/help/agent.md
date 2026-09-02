# PRA Agent

The right pane provides a persistent, RBAC-governed assistant for fleet operations. It can answer questions about engine health, drift, bundles, profiles, storage, and observability using Control Plane tools.

## Example questions

```text
Which engines are running Qwen3-4B?
Show deployments whose observed bundle differs from Registry intent.
Which engines have active storage alerts?
Summarize qualifications for the BALANCED profile.
```

The connection indicator reports whether the resumable WebSocket is active. Messages reconnect with a bounded backoff and replay from the last acknowledged sequence.

Tool activity is shown inline. Agent access does not bypass authorization: mutations still require the normal role, reason, confirmation, and audit path.

