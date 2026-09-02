# Control Plane Deployment

The reference stack is in `deploy/control-plane/compose.yaml`. Create a private
environment file, replace every placeholder, and start it with:

```bash
docker compose -f deploy/control-plane/compose.yaml up -d --build
```

Compose builds the Registry from `deploy/registry/Dockerfile` and starts it with
`pra-registry`. It builds the web application from
`deploy/control-plane/Dockerfile` and starts it with `pra-control`. The distinct
images share only documented REST contracts and PostgreSQL infrastructure; the
Control Plane does not import Registry persistence internals.

Terminate TLS at a reverse proxy and forward `Host`, `X-Forwarded-Host`, and
`X-Forwarded-Proto`. Set `public_url` to the external HTTPS origin. Do not expose
PostgreSQL, Tempo, Prometheus, or engine Management APIs to untrusted networks.

Back up the PostgreSQL database and configured durable Grafana volumes. A
release upgrade should take a database snapshot, deploy the new image, check
`/health`, verify SSO, inspect two engines, execute a harmless prefetch in a test
environment, and confirm both Control Plane and Registry audit records.

The current service creates its small owned schema on startup. Production
operators should pin image and package versions and treat schema changes as a
reviewed migration step as the early-access API evolves.
