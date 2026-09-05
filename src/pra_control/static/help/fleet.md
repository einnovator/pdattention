# Fleet and engines

Fleet joins live engine observations with desired Registry state. **Engine state** reports current reachability, while **Desired state** reports whether observed deployment matches Registry intent. A reachable engine can therefore have an unknown desired state without being unhealthy.

If Registry access is interrupted, statically or manually configured engines remain usable and keep their observed engine state. Their desired state is shown as **UNKNOWN** until Registry access returns; Fleet loading and PRA Agent connectivity do not wait for that recovery.

## Find an engine

Open Fleet from the fixed navigation rail. Search by text or filter by engine type, model, and desired state. Select a column heading to sort; the active order remains visible. Selecting an engine opens one reusable detail tab and focuses it if it is already open.

The central workspace exposes these engine views:

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

Small information icons beside labels explain metrics and settings in context. The explanation closes when you change tabs, dismiss it, or click outside it.
