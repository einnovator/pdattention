# Registry Deployment

## Local SQLite

```yaml
registry:
  host: 127.0.0.1
  port: 9200
  database_url: sqlite:///./pra-registry.db
  auth:
    mode: none
```

```bash
alembic -c alembic.ini upgrade head
pra-registry serve --config registry.yaml
```

`pra registry serve --config registry.yaml` is the equivalent discoverable
alias in the main PRA CLI. Production packaging uses the dedicated
`deploy/registry/Dockerfile` and `einnovator/pra-registry` image rather than the
commercial Control Plane image.

## PostgreSQL

Use PostgreSQL for shared deployments and run Alembic before each compatible
service rollout. The supplied `deploy/registry/docker-compose.yml` starts both.

```bash
docker compose -f deploy/registry/docker-compose.yml up --build
```

`deploy/registry/docker-compose.runtime-example.yml` shows a gateway using
`PRA_REGISTRY_URL=http://pra-registry:9200` and an environment-backed token.
Mount the runtime's `.pra/instances` directory on persistent storage so a
generated identity survives container replacement.

Environment variables override YAML: `PRA_REGISTRY_HOST`,
`PRA_REGISTRY_PORT`, `PRA_REGISTRY_DATABASE_URL`,
`PRA_REGISTRY_AUTH_MODE`, and `PRA_REGISTRY_TOKEN`.

Observability is off by default. Enable Prometheus and OTel explicitly in the
registry configuration. Back up PostgreSQL normally; external bundle bytes do
not need to be copied because their immutable locators and digests are stored.
