# Engine Deployment

Engine choice determines which PRA depth is available and economical. Prefix
caching, paged K/V, or a session cache does not by itself provide detached,
query-addressed Native Memory.

## Selection first

Every engine can benefit from Selected Context if it accepts ordinary text.
Measure that path before modifying attention or cache internals. For many
one-shot workloads it remains the best deployment.

## Native qualification

A Native Memory adapter must prove:

1. exact model/tokenizer/layer geometry;
2. correct positional and mask behavior;
3. parity or bounded quality change on held-out tasks;
4. no cross-request or cross-tenant influence;
5. exactly-once attachment and cleanup;
6. cold and warm economics against the same Selected Context.

Native Serving adds scheduler ownership of placement, prefetch, sharing,
eviction, and batching. It needs concurrency and tail-latency evidence in
addition to model correctness.

## Current support

Use the generated [engine support matrix](../engines/overview.md). Each engine
page identifies the recommended path, measured evidence, explicit missing
metrics, and production boundary.
