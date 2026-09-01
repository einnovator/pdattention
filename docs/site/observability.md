# Observability

PRA joins one request across four logical services:

```text
PRA Agent -> PRA Gateway -> PRA Runtime -> Inference engine
     |             |              |              |
     +-------------+--------------+--------------+
                   W3C trace context
```

Observability is **off by default**. With no `observability` section PRA does
not import an exporter, start a worker or metrics listener, hash identifiers,
or construct telemetry-only attributes. Primary qualification runs also keep
telemetry off unless the run explicitly measures observability overhead.

PRA exposes two complementary views:

| View | Answers |
| --- | --- |
| Distributed traces | Where did one request spend time across the Agent, Gateway, runtime, storage, and engine? |
| Operational metrics | What are rates, tails, errors, active sessions, and bytes over time? |
| Engine-native metrics | What did the engine scheduler, prefix cache, device, or physical K/V pool do? |
| PRA semantic metrics | What was selected, already visible, newly materialized, prefix-reused, native-reused, promoted, or evicted? |

PRA reuses engine-native telemetry when available and adds normalized wrapper
metrics where it is absent. It does not duplicate every engine metric under a
PRA name.

## Enable a profile

Install the optional libraries:

```bash
pip install -e ".[observability]"
```

Use the checked-in example:

```bash
pra gateway serve --backend vllm --backend-url http://localhost:8000 \
  --observability deploy/observability/observability.example.yaml
```

Or enable one backend explicitly:

```bash
pra gateway serve --backend ollama --backend-url http://localhost:11434 \
  --prometheus --prometheus-port 9464
```

Precedence is CLI overrides, explicit observability file, conventional OTel
environment variables, then disabled defaults. Prometheus and tracing are
independent gates.

## Privacy

The default `content.capture: none` excludes prompts, messages, documents,
tool arguments/results, credentials, and raw resource URIs. Traces may carry
short hashes for tenant, session, and task correlation. Prometheus labels are
limited to bounded values such as engine, model family, profile, execution
mode, status, record type, and storage tier. Request/session/user/resource IDs
never become metric labels.

Production deployments should put authentication and TLS in front of metrics,
Grafana, and collectors. All supplied Compose ports bind localhost.

## Next

- [OpenTelemetry](observability/opentelemetry.md)
- [Prometheus](observability/prometheus.md)
- [Grafana](observability/grafana.md)
- [Docker Compose](observability/docker-compose.md)
