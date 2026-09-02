# REST API Overview

All stable management routes are below `/v1/pra`. Responses are JSON and the
protocol identity returned by health and state endpoints is
`pra-management/1`.

## Read endpoints

| Method | Path | Purpose | Scope |
| --- | --- | --- | --- |
| `GET` | `/health` | Process/protocol health | `pra:read` |
| `GET` | `/info` | Engine instance and topology | `pra:read` |
| `GET` | `/capabilities` | Mechanism and qualification state | `pra:read` |
| `GET` | `/config` | Redacted effective and desired config | `pra:read` |
| `GET` | `/state` | Compact aggregate snapshot | `pra:read` |
| `GET` | `/models` | Loaded models; optional global `model_id` filter | `pra:models` |
| `GET` | `/models/{runtime_model_id}` | One model by engine-local identity | `pra:models` |
| `GET` | `/models/{runtime_model_id}/storage` | Model-scoped tier state | `pra:storage` |
| `GET` | `/profiles[/{name}]` | Effective profiles | `pra:read` |
| `GET` | `/resources[/{id}]` | Privacy-safe storage resources | `pra:read` |
| `GET` | `/sessions[/{id}]` | Privacy-safe sessions | `pra:sessions` |
| `GET` | `/storage` | Tier usage and lifecycle counters | `pra:storage` |
| `GET` | `/observability` | Telemetry state and links | `pra:read` |
| `GET` | `/metrics-link` | Prometheus link | `pra:read` |
| `GET` | `/trace-link` | Tempo/Grafana links | `pra:read` |
| `GET` | `/audit` | Recent local action audit | `pra:admin` |

Optional `POST /actions/load-model` and `POST /actions/unload-model` routes are
present in the common contract. Engines without real dynamic lifecycle support
return `501 ACTION_NOT_SUPPORTED` rather than simulating a model change.

Paths in this table are relative to `/v1/pra`. Collections support `offset`
and `limit`; resources additionally filter by `resource_type` and
`storage_tier`, while sessions filter by `status`.

## Response conventions

A collection response has stable pagination metadata:

```json
{
  "items": [],
  "total": 0,
  "offset": 0,
  "limit": 50,
  "next_offset": null
}
```

Structured errors identify a machine-readable code:

```json
{
  "error": {
    "code": "RESTART_REQUIRED",
    "detail": "Requested fields cannot be changed safely while the engine is running.",
    "restart_fields": ["device"]
  }
}
```

The API is versioned independently from the model inference protocol. Breaking
changes require a new URL prefix and management protocol identifier.
