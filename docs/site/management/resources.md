# Resources

Management responses deliberately expose metadata rather than context content
or native tensors.

## Engine and capability state

`EngineInstance` identifies the local process, engine/version, PRA version,
start time, accelerator summary, loaded models, and worker topology.
`EngineCapabilities` reports selected context, typed transport, native memory,
native serving, prefix cache, session-aware realization, storage tiers, and
observability independently.

Capability status uses `AVAILABLE`, `VALIDATED`, `CANDIDATE`, `RESEARCH_ONLY`,
`BLOCKED`, `NOT_MEASURED`, or `NOT_APPLICABLE`. Mechanism availability does not
silently imply validated quality, favorable economics, or a product
recommendation.

## Models and profiles

Loaded-model rows contain immutable model/tokenizer fingerprints when known,
quantization, device placement, PRA bundle identity, profile, execution mode,
load time, runtime state, and an engine-local `runtime_model_id`. Profile rows identify their source, effective
policy, qualification status, and whether they are immutable or centrally
managed.

Model-native storage, session handles, bundles, qualification, and effective
configuration are scoped to that runtime ID. Native identity additionally
includes the model fingerprint, resource version, and selected representation;
K/V produced by one model is never reused by another. Canonical SOURCE records
may be shared because they contain no model-native representation.

## Stored resources

Resource IDs are one-way, process-local privacy-safe identifiers. A row may
include type, version, byte/token counts, tier, native residency, pin count,
last access, scope summaries, and a checksum. It never includes source text,
tool payloads, credentials, source URIs, tenant IDs, task IDs, session IDs, or
K/V values.

## Sessions

Session IDs are also privacy-safe hashes. Summaries report activity, task count,
visible-context counts, selected/reused token totals, and only Boolean engine
cache-handle presence. They include `runtime_model_id` and model fingerprint so
a model switch can invalidate incompatible native and prefix state. Prompt messages, result bodies, opaque cache handles,
tenant identities, and worker identities are excluded.

## Storage and observability

Storage state keeps HOT/WARM/COLD/SOURCE bytes and counts separate from quotas,
evictions, reloads, promotions, reconstruction counts, retention policy, and
maintenance status. Observability state reports enablement and configured links;
it does not proxy arbitrary telemetry queries.
