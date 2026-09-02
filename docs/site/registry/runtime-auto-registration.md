# Runtime Auto-Registration

Managed engines and gateways can advertise themselves to the PRA Registry. The
Registry becomes the fleet directory and desired-state authority; each runtime
remains authoritative for its observed state.

```text
engine or gateway starts
        -> management listener becomes ready
        -> stable identity is loaded or created
        -> full observed state is registered
        -> compact heartbeats maintain liveness
        -> changed state is published
        -> Control Plane discovers the instance
```

Model, bundle, and deployment records are not runtime instances. A
`ManagedInstance` has type `ENGINE` or `GATEWAY`, placement labels, management
and inference URLs, versions, capabilities, loaded-model summaries,
observability links, desired/observed revisions, and liveness timestamps. Raw
credentials and context/session contents are never accepted in this resource.

## Engine configuration

```yaml
management_api:
  enabled: true
  host: 0.0.0.0
  port: 9101
  registry:
    enabled: true
    url: https://pra-registry.example.com
    required: false
    heartbeat_seconds: 30
    refresh_seconds: 300
    auth:
      type: bearer
      token_env: PRA_REGISTRY_TOKEN
    instance:
      id: prod-vllm-01
      name: Production vLLM 01
      environment: production
      region: west
      cluster: inference-west
      namespace: llm
      host: prod-vllm-01.internal
      management_url: https://prod-vllm-01.internal:9101
      inference_url: https://prod-vllm-01.internal:8000
      labels:
        team: platform
```

The gateway uses the same `registry` schema under
`gateway.management_api`. Its `inference_url` points to the gateway, while its
observed model list summarizes upstreams rather than claiming engine-local
model residency.

## Stable identity

Identity resolution is deterministic:

1. Explicit `instance.id`.
2. The persisted identity file.
3. A generated UUID written atomically to the PRA home directory.

The default files are `.pra/instances/engine.json` and
`.pra/instances/gateway.json` under the configured PRA home. A process ID is
observability data, not instance identity. Set `instance.identity_file` when a
container mounts identity storage elsewhere.

## Availability behavior

Registration is optional unless `required: true`. With the default resilient
mode, Registry outage does not stop inference: exponential backoff retries in
the background and the management endpoint reports the last error. Strict mode
fails startup when the initial registration cannot be established.

A heartbeat contains health, uptime, observed revision, and a compact runtime
summary. Full capabilities, model summaries, and observability URLs are sent
on startup, periodically, and after supported state changes. The Registry marks
an instance `OFFLINE` when its heartbeat exceeds `offline_after_seconds`; a
clean shutdown deregisters immediately.

```bash
pra engine registry-status --management-url http://engine:9101
pra engine register --management-url http://engine:9101
pra gateway registry-status --management-url http://gateway:9150
pra gateway register --management-url http://gateway:9150
pra registry instances --type ENGINE --environment production
pra registry offline
```

## Credentials

Bearer tokens can come from an environment variable or secret file. OAuth2
client credentials use environment-backed client ID/secret references and a
token URL. mTLS uses local certificate, key, and optional CA paths. Payloads
contain only the authenticated Registry subject as `credential_identity`; they
never contain a token, password, private key, or client secret.
SDK embedders can inject a `token_provider` callback for an external secret
manager without placing its returned value in runtime configuration.

## Desired-state drift

`GET /v1/instances/{id}/desired` resolves an applicable deployment by
environment, cluster, name, host, and labels. The runtime records differing
model, bundle, profile, and mode fields but does not automatically unload a
model, change topology, or perform another destructive action. Those remain
explicit management or orchestrator operations.

## Multi-host development

Use routable advertised URLs, even if listeners bind to all interfaces:

```yaml
registry:
  url: http://registry.lab:9200
  instance:
    host: mlx-01.lab
    management_url: http://mlx-01.lab:9101
    inference_url: http://mlx-01.lab:8000
```

Protect non-loopback services with private networking and authentication. The
Control Plane's `fleet.discovery_mode: registry` then needs no static engine
list; it reads online `ENGINE` instances directly from `/v1/instances`.
