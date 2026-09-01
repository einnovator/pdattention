# PRA federated observability stack

Observability is off unless both the runtime configuration and the Compose
profile are enabled. Start only the collector, Prometheus, and Grafana:

```bash
OBSERVABILITY_BIND_ADDRESS=0.0.0.0 \
GRAFANA_ADMIN_PASSWORD='choose-a-password' \
docker compose --profile observability up -d
```

Then start PRA on the host, which is the normal path for Hugging Face, MLX,
AirLLM, and FreeToken:

```bash
pra gateway serve --backend vllm --backend-url http://localhost:8000 \
  --observability deploy/observability/observability.example.yaml
```

Grafana is at `http://<collector-host>:3000`, Prometheus at port `9090`, Tempo
at port `3200`, and OTLP accepts gRPC on port `4317` or HTTP/protobuf on port
`4318`. The default bind remains loopback. Setting
`OBSERVABILITY_BIND_ADDRESS=0.0.0.0` makes the lab stack reachable from other
machines and must only be used on a trusted network or behind TLS and access
control.

## Federated engines

Copy `prometheus-targets/engines.example.json` to an environment-specific JSON
file and list every engine metrics endpoint. Prometheus reloads these files
without a restart. Each engine process must bind its metrics endpoint to an
address reachable from the collector host. Point its OTLP exporter at the same
collector:

```bash
python engine_telemetry_probe.py \
  --engine mlx --model Qwen3-4B --machine mac-48gb \
  --engine-url http://127.0.0.1:8000/health \
  --otlp-endpoint http://192.168.1.102:4317 \
  --metrics-port 9464
```

The probe is the compatibility path for engines without native OTel. It checks
the live engine endpoint and emits real PRA spans and normalized metrics. Keep
engine-native exporters enabled as separate scrape jobs when they expose
scheduler, cache, GPU, or engine-specific counters.

Grafana provisions two views for every supported engine:

- `PRA + <engine>` reads normalized Prometheus metrics.
- `PRA + <engine>: OTEL Traces` reads request and lifecycle spans from Tempo.

Resource attributes `pra.engine`, `pra.model_family`, `host.name`, and
`machine.role` distinguish engines and machines in the shared trace store.

Containerized engine examples are overlays. Compose must include both files so
the engine can resolve the collector service:

```bash
MODEL=Qwen/Qwen3-0.6B \
docker compose -f docker-compose.yml -f docker-compose.engines.yml \
  --profile observability --profile vllm up -d
```

Profiles are available for `vllm`, `sglang`, `openvino`, `tensorrt-llm`,
`llama-cpp`, and `ollama`. MLX/Metal cannot run usefully in Linux containers;
run MLX on the macOS host and point it at the container collector. AirLLM,
FreeToken, and the embedded HF runtime use the same hybrid-host arrangement.
Engine-native detailed tracing remains separately gated by
`engine_native.enable_tracing`.
