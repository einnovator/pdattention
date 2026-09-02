# Gateway Management API

The open-source **PRA Gateway Management API** reports and safely changes one
gateway's upstream routing, capability negotiation, session transport, and
fallback policy. It is distinct from both the inference API and the
[Engine Management API](index.md).

| Property | Value |
| --- | --- |
| Protocol | `pra-gateway-management/1` |
| API prefix | `/v1/pra/gateway` |
| Default state | Disabled |
| Default bind | `127.0.0.1:9150` |
| Inference listener | Separate, normally `:8080` |
| OpenAPI | `/openapi.json` |
| Swagger UI | `/docs` |
| ReDoc | `/redoc` |

The checked-in contract is available as
[PRA Gateway Management OpenAPI](../api/openapi/pra-gateway-management-v1.json).

The gateway manages logical transport and upstream choice. Engine-local
scheduler, model, and physical K/V state remain behind each engine's management
API. Native K/V never crosses this REST interface.

## Enable the listener

The listener, OpenAPI application, authentication backend, and background
registration are not created by default. Enable them explicitly:

```bash
pra gateway serve \
  --mode typed-transport \
  --backend vllm \
  --backend-url http://127.0.0.1:8000/v1 \
  --management-api \
  --management-port 9150
```

The inference and management listeners then remain independently isolatable:

```text
http://127.0.0.1:8080/v1/chat/completions       inference
http://127.0.0.1:9150/v1/pra/gateway/health    management
http://127.0.0.1:9150/docs                     Swagger
```

Equivalent YAML is:

```yaml
gateway:
  management_api:
    enabled: true
    host: 127.0.0.1
    port: 9150
    auth:
      mode: static_bearer
      token_env: PRA_GATEWAY_MANAGEMENT_TOKEN
```

`PRA_GATEWAY_MANAGEMENT_ENABLED`, `PRA_GATEWAY_MANAGEMENT_HOST`,
`PRA_GATEWAY_MANAGEMENT_PORT`, `PRA_GATEWAY_MANAGEMENT_AUTH_MODE`, and
`PRA_GATEWAY_MANAGEMENT_TOKEN` are explicit environment overrides. An
unauthenticated listener may bind only to loopback.

## Resource model

| Resource | What it describes | Sensitive-data rule |
| --- | --- | --- |
| Gateway instance | Identity, version, process, health, environment, observability, Registry state | No credentials |
| Upstream | URL, engine kind, models, health, priority, weight, capability handshake | Credential reference only |
| Session | Hashed session and tenant IDs, counts, prefix digest, transport and reuse | No raw messages |
| Resource knowledge | Hashed identity, version, type, size, acknowledgement state | No resource body |
| Transport | Effective transport and byte, fallback, reconnect, resync, and reuse counters | Aggregates only |
| Policy | Upstream selection, fallback, affinity, retry, authorization, observability | Secrets excluded |

Session and resource identifiers are one-way hashes. This makes the API useful
for operations without turning it into a transcript or document export API.

## Read endpoints

| Method and path | Result |
| --- | --- |
| `GET /v1/pra/gateway/health` | Liveness and protocol identity |
| `GET /v1/pra/gateway/info` | Gateway identity and registration state |
| `GET /v1/pra/gateway/capabilities` | Effective gateway and upstream capabilities |
| `GET /v1/pra/gateway/config` | Redacted effective configuration |
| `GET /v1/pra/gateway/state` | Combined gateway, policy, session, resource, and transport summary |
| `GET /v1/pra/gateway/upstreams` | Paginated upstream inventory |
| `GET /v1/pra/gateway/upstreams/{id}` | One upstream and its latest handshake |
| `GET /v1/pra/gateway/sessions` | Paginated privacy-safe sessions; supports `model` filtering |
| `GET /v1/pra/gateway/sessions/{id}` | One session by returned hash |
| `GET /v1/pra/gateway/resources` | Paginated known resources; supports `record_type` filtering |
| `GET /v1/pra/gateway/transport` | Transport, delta, fallback, and visible-reuse counters |
| `GET /v1/pra/gateway/observability` | Prometheus, OTel, Grafana, and Tempo status |
| `GET /v1/pra/gateway/audit` | Newest-first bounded local audit log |

List endpoints accept `offset` and `limit`; `limit` is bounded to 200.

## Upstreams and routing

Create and safely update an upstream without returning its credential:

```http
POST /v1/pra/gateway/upstreams
Content-Type: application/json

{
  "upstream_id": "gpu-primary",
  "name": "Production vLLM",
  "base_url": "http://vllm:8000/v1",
  "management_url": "http://vllm:9101",
  "provider": "vllm",
  "models": ["Qwen/Qwen3-4B"],
  "auth_reference": "VLLM_SERVICE_TOKEN",
  "priority": 10,
  "weight": 1.0,
  "enabled": true
}
```

`auth_reference` names a server-side environment variable; its value is never
serialized. `PATCH /upstreams/{id}` changes bounded fields and
`DELETE /upstreams/{id}` removes an endpoint, except that the last endpoint
cannot be removed while the gateway is active.

The policy supports `static`, `model`, `capability`, `tenant`, `weighted`, and
`failover` selection. Session affinity pins subsequent turns to the chosen
upstream unless it is disabled. Scheduler decisions stay inside the upstream
engine.

## Negotiation and safe actions

```http
POST /v1/pra/gateway/actions/renegotiate/{upstream_id}
POST /v1/pra/gateway/actions/health-check/{upstream_id}
POST /v1/pra/gateway/actions/resync-session/{session_hash}
POST /v1/pra/gateway/actions/drop-session/{session_hash}
POST /v1/pra/gateway/actions/clear-capability-cache
POST /v1/pra/gateway/actions/reload-policy
```

Every action accepts a reason and optional idempotency key:

```json
{"reason": "upstream restarted", "idempotency_key": "restart-2026-09-02"}
```

Negotiation exposes selected-context, typed-transport, native-memory,
native-serving, storage-lifecycle, model-fingerprint, compatibility, expiry,
fallback, and rejection state. Resync invalidates uncertain engine handles and
forces reconstruction on the next turn. Drop closes both logical and upstream
session state.

Safe mutable gateway settings use `PATCH /v1/pra/gateway/config`. The accepted
surface is the default profile, gateway mode, fallback placement, and bounded
gateway policy. Arbitrary code execution, environment mutation, and credential
retrieval are intentionally absent.

## Authentication and audit

The listener supports localhost/no-auth development, static bearer tokens,
JWT/OIDC, and mTLS. Production scopes are:

| Scope | Access |
| --- | --- |
| `pra-gateway:read` | Read health, configuration, upstream, resource, and transport state |
| `pra-gateway:configure` | Patch or reload safe policy |
| `pra-gateway:sessions` | Inspect, resync, and drop sessions |
| `pra-gateway:upstreams` | Create, change, remove, probe, and renegotiate upstreams |
| `pra-gateway:admin` | All gateway-management operations and audit |

Mutations append actor, timestamp, request ID, optional trace ID, reason,
redacted before/after summaries, and success or failure. Send `X-Request-ID`
and `X-Trace-ID` to correlate automation with the audit log.

## CLI

All inspection commands work against a remote management URL:

```bash
pra gateway health --management-url http://gateway:9150
pra gateway inspect --management-url http://gateway:9150
pra gateway upstreams --management-url http://gateway:9150 --json
pra gateway sessions --management-url http://gateway:9150 --json
pra gateway transport --management-url http://gateway:9150 --yaml
pra gateway config --management-url http://gateway:9150 --yaml
pra gateway renegotiate gpu-primary --reason "engine upgraded" \
  --management-url http://gateway:9150
pra gateway resync SESSION_HASH --reason "stale capability state" \
  --management-url http://gateway:9150
```

Supply `--token` or `PRA_GATEWAY_MANAGEMENT_TOKEN` for bearer authentication.
Use `--json` or `--yaml` for automation.

## Observability

The gateway reuses the common Prometheus and OpenTelemetry pipeline. Its
management view includes requests, latency, active sessions, upstream errors,
fallbacks, resyncs, transport/message/resource/delta bytes, visible reuse, new
materialization, and capability-negotiation counters. URLs for Prometheus,
Grafana, and Tempo are links to their actual services, not proxy endpoints.

See [Observability](../observability.md) for exporter setup and dashboard
deployment.

## Registry and Control Plane

Optional Registry settings make the gateway register on startup, heartbeat its
health/capabilities/upstream summary, and mark itself offline on clean shutdown:

```yaml
gateway:
  management_api:
    registry:
      enabled: true
      url: http://registry:9200
      token_env: PRA_REGISTRY_TOKEN
      deployment_id: edge-gateway-1
      model_id: qwen3-4b
```

The open Registry is the system of record. The commercial eInnovator Control
Plane discovers and governs fleets by aggregating the public gateway and engine
management contracts; inference traffic does not need to pass through it.
