# Observability and PRA Agent

The Control Plane presents lightweight fleet signals and links to the existing
Grafana, Prometheus, and Tempo installations. It does not implement a second
metrics store. Engine links include an engine variable; trace links can include
a trace ID. Whether an iframe can embed a panel depends on the Grafana security
policy.

The right-side PRA Agent panel uses PRA SDK tool schemas for fleet capabilities.
Its toolbar can create and resume sessions, show command tips, and collapse the
chat area. The model selector exposes discovered runtime aliases together with
reachability. A configured OpenAI-compatible engine can answer from the
authorized fleet snapshot; model errors fall back to deterministic manager
answers. Tool calls enter the same fleet service and RBAC checks as UI requests.

The WebSocket protocol supplies:

- a durable resume token bound to the authenticated subject;
- monotonically increasing event sequence numbers;
- recent-event replay after reconnect;
- client message IDs and server-side duplicate suppression;
- ping/pong heartbeat and exponential client retry;
- streamed response deltas and visible tool status.

Browser commands mirror the common TUI surface for status, fleet summary,
model discovery and selection, sessions, clearing, and help. The PRAAgent TUI
also provides `/sessions` and `/tips`, and its `/models` output includes target
status.

The assistant is not an autonomous production controller. Optimization is
recommendation-only and mutations require the normal role, reason,
confirmation, and audit workflow.
