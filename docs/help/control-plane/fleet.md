# Fleet and engines

Fleet overview joins live engine observations with desired Registry state. Status distinguishes engines that are in sync, drifting, offline, or not yet classified.

## Find an engine

Use the Fleet filter in the left pane, then select an engine. The central workspace exposes these views:

| View | Purpose |
| --- | --- |
| Summary | Current model, bundle, profile, mode, health, and drift |
| Capabilities | Runtime-advertised PRA and serving capabilities |
| Config | Sanitized effective configuration |
| Models | Loaded or available model state |
| Sessions | Active session summaries and affinity |
| Resources | Registered context and native-memory resources |
| Storage | HOT, WARM, and COLD lifecycle state |
| Observability | Metrics, traces, and dashboard links |
| Audit | Engine-scoped action history |

## Engine actions

The action menu can prefetch, promote, demote, evict, or run maintenance. High-impact operations require explicit confirmation and every accepted request records the operator, reason, result, and trace identifier.

Open the chart button on an engine to follow its configured Grafana observability link.

