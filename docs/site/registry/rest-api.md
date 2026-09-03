# Registry REST API

The protocol identifier is `pra-registry/1`. The service exposes OpenAPI,
Swagger, and ReDoc at `/openapi.json`, `/docs`, and `/redoc`.

| Surface | Endpoints |
| --- | --- |
| Models | `GET/POST /v1/models`, `GET/PATCH/DELETE /v1/models/{id}` |
| Bundles | `GET/POST /v1/bundles`, `GET/PATCH /v1/bundles/{id}`, approve/deprecate actions |
| Profiles | `GET/POST /v1/profiles`, `GET/PATCH /v1/profiles/{id}`, approve action |
| Compatibility | `GET/POST /v1/compatibility`, `GET /v1/compatibility/resolve` |
| Qualifications | `GET/POST /v1/qualifications`, `GET /v1/qualifications/{id}` |
| Deployments | `GET/POST /v1/deployments`, `GET/PATCH /v1/deployments/{id}`, desired state |
| Policies | `GET/POST /v1/policies`, `GET/PATCH /v1/policies/{id}`, approve action |
| Governance | `GET/POST /v1/approvals`, `GET /v1/audit` |
| Resolution | `POST /v1/resolve/bundle`, `/profile`, and `/deployment` |
| Managed instances | Register, heartbeat, observed update, deregister, list/get, and desired-state pull under `/v1/instances` |

Lists use bounded `limit`/`offset` pagination and indexed filters. Resolver
ordering is total and deterministic: approval, exact revision, qualified engine
recommendation, stable resource ID, then immutable revision.

Qualification creation requires a canonical `condition` ID in addition to its
display `mode`. Bundle conditions also require an exact `bundle_id` and
`bundle_revision`. See [PRA Execution Conditions](../execution-conditions.md).

```bash
curl -s http://127.0.0.1:9200/v1/models?limit=50
curl -s -X POST http://127.0.0.1:9200/v1/resolve/bundle \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-14B","engine":"vllm"}'
```

The checked-in schema is [PRA Registry OpenAPI](../api/openapi/pra-registry-v1.json).
