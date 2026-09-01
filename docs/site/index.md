# Progressive Retrieval Attention

PRA is context infrastructure for AI agents and long-running model
applications. It keeps documents, tools, results, task state, and other reusable
context addressable outside the active prompt, then selects and materializes
only what each model operation needs.

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

## What PRA changes

### Send less

**Selected Context** chooses relevant evidence and presents it through an
ordinary model input. It works with unmodified inference engines and is the
recommended starting point.

### Recompute less

**Native Memory** keeps qualified, selected resources in model-native form so
repeated requests can reuse them without reconstructing the same visible text.
It is useful only when model parity, quality, and workload economics have been
measured.

### Manage reuse explicitly

**Native Serving** lets a scheduler own placement, prefetch, sharing, eviction,
and batching for native resources. This is the deepest integration and the most
engine-specific one.

## Start here

- [Getting Started](getting-started.md): evaluate PRA in five minutes or connect
  an existing application.
- [CLI](cli.md): run the complete qualification journey and export an assessment.
- [Model Support](models.md): find built-in mappings and adapter requirements by family.
- [Web UI](web-ui.md): operate local agent sessions, tasks, records, and approvals in a browser.
- [Concepts](concepts.md): understand logical context, active context, routing,
  and materialization.
- [Engine Support](engines/overview.md): choose the deployment supported by your
  engine today.
- [Metrics & Qualification](metrics.md): compare full, selected, native, and
  serving paths without mixing retrieval with transport.
- [Deployment](deployment/index.md): choose an agent, gateway, SDK, or native
  engine path.

## Product boundary

Typed PRA Transport preserves record identity, provenance, task/session scope,
authorization, deltas, and explicit fallback. It does not send raw K/V over the
application protocol. Native Memory is an engine capability negotiated below
that transport boundary.

!!! note "Evidence before depth"
    A deeper integration is not automatically faster or better. Start with
    Selected Context, freeze the selected evidence, and qualify each deeper
    mechanism against the same workload.

Paper-specific terminology, experimental variants, and raw artifact paths are
kept in [Research / Evidence](research/index.md), away from the first-use path.
