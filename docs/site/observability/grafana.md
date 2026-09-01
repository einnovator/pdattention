# Grafana

The repository provisions Prometheus and Tempo datasources and 24 dashboards from
`deploy/observability/grafana/`.

Baseline views cover:

- multi-engine overview;
- PRA Agent;
- Gateway;
- runtime and "Why PRA helped" context economics;
- HF, vLLM, SGLang, MLX, OpenVINO, TensorRT-LLM, AirLLM, llama.cpp, Ollama,
  and FreeToken.

Each engine has two explicit dashboards. `Prometheus Metrics` shows normalized
PRA request, context, native-memory, and storage measurements, plus available
engine-native panels. `OTEL Traces` searches Tempo for that engine's request
and lifecycle spans. The resource attributes distinguish model and host even
when several machines write to one collector.

Every dashboard supports datasource, environment, service, engine, model,
host, instance, profile, and execution-mode variables. No dashboard hard-codes
an engine host. Engines without a native telemetry surface still show PRA
wrapper metrics; unavailable native panels are not fabricated.

```bash
cd deploy/observability
docker compose --profile observability up -d
```

Open `http://localhost:3000` for a local stack or
`http://<collector-host>:3000` for a federated lab. The overview correlates context reduction,
visible/native reuse, storage residency, request latency, and successful
throughput. This separates a routing benefit from prefix-cache reuse or a
storage-promotion penalty.

Dashboard JSON is generated deterministically:

```bash
python deploy/observability/generate_dashboards.py
```

Do not store credentials in dashboard JSON. Configure authentication, TLS,
alerting, and retention in the deployment environment.
