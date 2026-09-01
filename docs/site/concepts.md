# Concepts

PRA separates what an application *knows* from what a model operation needs
*right now*. This page builds that model from the outside in.

```text
large logical context
    |
typed records + addresses + policy
    |
route -> select -> materialize
    |
small active working context
    |
model
```

## 1. Logical context

Logical context is the complete context available to an application: documents,
conversation history, task state, tool definitions, tool results, files, and
other reusable resources. It can be much larger than one model input.

Logical context has stable identity and lifecycle. A record can remain available
without being visible to every model call.

## 2. Active context

Active context is the bounded working set supplied to one model operation. PRA
constructs it from ordinary messages plus selected views or selected native
memory. Keeping this distinction explicit prevents a long-running session from
becoming an ever-growing prompt.

## 3. Typed record

A typed record stores identity, kind, version, provenance, authorization scope,
task/session ownership, and one or more views of exact backing data. Examples
include documents, tool descriptions, skills, API results, logs, tables, and
terminal output.

The record type guides compaction and materialization. A table can expose a
schema and selected rows; a log can expose a time range; a tool can expose a
compact selection description before its complete schema.

## 4. Compact selection and address views

Routing should not require the full backing payload. PRA builds compact views
containing stable IDs and enough lexical, semantic, structural, or typed
information to address useful regions. These views are cheap to keep active and
do not replace the exact source.

## 5. Routing and selection

Routing scores compact views against the current query. Selection applies
budgets, authorization, task scope, and optional diversity constraints to choose
record IDs and intervals. A selector decides *what* is relevant; it does not
decide how the engine will consume the result.

## 6. Materialization

Materialization resolves selected IDs into a concrete working representation:

- labeled text for an ordinary engine;
- typed resource bodies for a PRA-aware endpoint;
- selected, model-specific native memory for a qualified engine.

The selected identities and intervals should remain frozen when comparing these
representations.

## 7. Exact backing detail

Compaction never becomes the authoritative source. Exact backing detail remains
hash-verifiable and recoverable from memory, durable storage, or a source
loader. A selected row, span, or object can be reconstructed without presenting
the entire record.

## 8. Session and task scope

Sessions bind records to a user, tenant, model, and long-running interaction.
Tasks add status and dependencies. Scope affects visibility, retention, and
selection; relevance never grants authorization. Closing a model session can
release ephemeral native state without deleting the logical record stream.

## 9. Profiles

A profile is a measured combination of model, engine, consumer layers, storage
policy, and workload assumptions. Profile names are not universal quality
claims. `BALANCED` currently retains all eligible consumer layers unless a
model-specific held-out calibration says otherwise. Candidate profiles remain
explicitly pending.

## 10. Storage lifecycle

Native memory is derived data. PRA can keep it attention-ready, retain a
lossless warm representation, persist a colder representation, or reconstruct
it from the authoritative source. Quotas, authorization, request pins, task
state, and reuse govern transitions.

## 11. Three deployment depths

| Depth | What crosses the boundary | Engine changes | Best first use |
| --- | --- | --- | --- |
| Selected Context | Selected ordinary text | None | Broad compatibility |
| Typed PRA Transport | Typed records, IDs, deltas, policy | Gateway or aware endpoint | Agents and durable sessions |
| Native Memory | Selected model-native memory | Model/engine integration | Repeated immutable resources |
| Native Serving | Placement and lifecycle hints | Scheduler integration | Shared, concurrent serving |

These depths compose, but they do not form a quality ranking. The best deployment
is the shallowest one that satisfies the measured workload.
