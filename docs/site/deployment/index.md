# Choose a Deployment Path

Use the shallowest integration that satisfies the measured workload.

```text
Need zero engine changes?
 -> Selected Context / Gateway

Need typed resources end to end?
 -> Typed PRA Transport / SDK

Repeated immutable resources and selected-text prefill is expensive?
 -> qualify Native Memory

Scheduler owns placement and prefetch?
 -> qualify Native Serving
```

> The deepest integration is not automatically the best integration.

## Decision table

| Need | Start with | Why |
| --- | --- | --- |
| Existing OpenAI-compatible application | [Gateway](gateway.md) | No application protocol replacement |
| Durable tools, results, tasks, and sessions | [Agents](../agents/index.md) | Typed state and authorization stay explicit |
| Embedded Python control | [Runtime / SDK](runtime-sdk.md) | Direct lifecycle and trace access |
| Repeated immutable evidence | [Engine qualification](engines.md) | Native reuse may amortize encoding |
| Concurrent shared native resources | Native Serving qualification | Scheduler must own safe physical reuse |

## Promotion sequence

1. Establish Full Context and Selected Context quality with a frozen selector.
2. Add Typed PRA Transport if identity, deltas, tasks, or authorization must
   survive the application boundary.
3. Compare the same selected evidence as text and Native Memory.
4. Add scheduler-managed Native Serving only after isolation, lifecycle, and
   queueing tests pass.

Each promotion needs its own rollback path and evidence receipt. Fallback must
be explicit; a request requiring native behavior must not silently become text.

See [Metrics & Qualification](../metrics.md) and the current [engine support
matrix](../engines/overview.md).
