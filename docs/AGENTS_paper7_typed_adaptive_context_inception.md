# Paper 7: Typed Adaptive Context

## Thesis

Represent real agent context as typed, stateful, addressable records with
separate discovery/address views, compact visible views, lossless backing
originals, selective/adaptive materialization, and optional stateful cursors.

Paper 7 builds on the Paper 3.1 summary-index analysis and Paper 6.5 typed
tool/skill records. It is a functional unification layer, not a claim that one
compression or retrieval policy solves every agent workload.

## Result records

Support `TOOL_RESPONSE`, `LOG_BLOCK`, `TERMINAL_OUTPUT`, `DB_RESULT`,
`GRAPH_RESULT`, `RAG_RESULT`/`RAG_CHUNK_SET`, `FILE_READ`, `API_RESULT`, and
`GENERIC_TEXT`. Keep Paper 6.5 `TOOL` and `SKILL` capability records.

Each result record provides:

1. metadata/identity;
2. a bounded compact view;
3. independent lexical, entity, rare-term, schema, summary, dense, or native-QK
   address views as available;
4. exact scoped backing state;
5. selected-field/range and full materialization.

## Storage and transport

The default local backing store must provide content-addressed identities,
exact originals, hashes, provenance, tenant/session scope, TTL, size bounds,
cleanup, and optional persistence. A handle is not authorization.

Support `upfront`, `on_demand`, and default `adaptive` transport policies for
same-process, separate local-process, remote-model, and distributed-store
topologies. Thresholds such as 100K and 1M are experiment parameters, not
universal claims. Measure bytes, latency, round trips, active K/V, and reuse.

## Expansion modes

Compare:

- native `MATERIALIZE(record_id, selector?, cursor?)` events;
- explicit `retrieve_record`, `search_record`, and `fetch_cursor` tools;
- mixed bounded response plus cursor;
- proactive/adaptive expansion.

Exact identities must be resolved without semantic rediscovery.

## Stateful cursors

`CursorRecord` stores cursor and source identity, query, schema, collection,
position/range, page size, filters/order, total estimate, TTL, provenance,
authorization, and continuation handle. Support next/previous, range,
search/filter, aggregate, sample, selected fields, and close.

Target database and graph analytics, Graph RAG, large search results,
paginated APIs, and time series. Include drill-down tasks such as aggregate to
subgroup, anomaly to rows, graph-neighborhood expansion, and aggregate to
exemplars.

## Addressability benchmark

Plant low-salience entities, aliases, numbers/dates, rare strings, relations,
and action triggers in tool outputs, logs, database rows, and RAG results.
Separate:

- explicit insufficiency, where the query names the missing concept;
- latent insufficiency, where the hidden detail is needed to infer an action;
- action-trigger omission, where the hidden detail changes the next tool/action.

Primary metric:

`TriggerRecall = P(required downstream action remains reachable | compact view)`.

Measure compact answer/evidence, expansion, next action/tool, whether the model
knew to ask, and whether proactive/indexed retrieval repaired an omission.
Do not infer end-to-end model competence from deterministic mechanism checks.

## Baseline and falsification

Implement a Headroom-inspired architectural baseline: type-aware compression,
local original cache, marker/handle, model retrieval tool, and optional
proactive expansion. Compare mechanisms rather than claiming implementation
identity or product failure.

- Savings with collapsed trigger recall means compression is not a safe default.
- Tool retrieval matching native events makes native events optional.
- Cursors that do not improve large-data tasks require narrow claims.
- Proactive expansion being necessary means model-emitted retrieval is insufficient.
- Address views that do not repair omission do not solve CCR brittleness.

## Milestones

- M0: record/store correctness.
- M1: type-specific compressors.
- M2: explicit recovery.
- M3: latent/action-trigger benchmark.
- M4: native event versus tool versus mixed cursor versus proactive expansion.
- M5: database/graph analytics cursors.
- M6: adaptive storage/transport.
- M7: end-to-end agent workloads.

Avoid a full Cartesian product. Keep mechanism and model-dependent findings
separate.

## Security

Authorize before expansion or page fetch. Scope cursors to tenant, session, and
call; enforce TTL and revocation; retain audit events; apply redaction policy to
both compact and full views.

## Deliverables

Implementation, public record/store/cursor APIs, local/remote transport policy,
Headroom-style baseline, benchmarks, figures/tables, reproducibility manifest,
tests, paper source/PDF, and visual PDF QA.
