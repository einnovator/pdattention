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

To populate both dashboards with bounded generation traffic rather than health
probes, run the engine telemetry probe against an OpenAI-compatible chat endpoint:

```bash
python deploy/observability/engine_telemetry_probe.py \
  --engine llamacpp --model qwen2.5-0.5b \
  --engine-url http://127.0.0.1:18080/v1/chat/completions \
  --queries deploy/observability/dataset_queries.example.json \
  --otlp-endpoint http://COLLECTOR:4317 --metrics-port 9464
```

The example cycles QASPER-, HotpotQA-, and 2Wiki-style requests. Tempo receives
`pra.engine.request` spans with a bounded `pra.dataset` attribute; Prometheus
receives request latency and context counters. Use health-only probing only for
availability dashboards.

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
