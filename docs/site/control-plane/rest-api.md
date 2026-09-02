# Control Plane REST API

REST is a thin adapter over `ControlManager`. Existing browser endpoints remain
under `/api`; OpenAPI is available at `/docs` and `/openapi.json`.

## Semantic exposure

```yaml
control_plane:
  rest:
    enabled: true
    allow: ["fleet.*", "engine.inspect", "qualification.*", "action.plan"]
    deny: ["action.apply"]
```

Patterns match operation IDs, not URLs. Deny wins. Disabled operations are not
registered, so they return `404` and disappear from OpenAPI.

## Plan example

```bash
curl -X POST \
  'http://127.0.0.1:9300/api/actions/plan?action=prefetch&target=mlx-01' \
  -H 'Content-Type: application/json' \
  -H 'X-CSRF-Token: ...' \
  -H 'Idempotency-Key: launch-42' \
  -d '{"values":{"resource_id":"document-42"},"reason":"prepare launch"}'
```

Applying the returned plan uses
`POST /api/actions/plans/{plan_id}/apply`. Browser mutations require CSRF;
manager permissions and audit apply regardless of transport.

Domain failures use a stable envelope:

```json
{"error":{"code":"approval_required","message":"evict requires confirmation","details":{}}}
```
