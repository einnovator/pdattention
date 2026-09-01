# PRA observability stack

Observability is off unless both the runtime configuration and the Compose
profile are enabled. Start only the collector, Prometheus, and Grafana:

```bash
docker compose --profile observability up -d
```

Then start PRA on the host, which is the normal path for Hugging Face, MLX,
AirLLM, and FreeToken:

```bash
pra gateway serve --backend vllm --backend-url http://localhost:8000 \
  --observability deploy/observability/observability.example.yaml
```

Prometheus is at `http://localhost:9090`, Grafana at
`http://localhost:3000`, and OTLP accepts gRPC on `localhost:4317` and HTTP on
`localhost:4318`. All published ports bind loopback. Put authentication and
TLS in front of them for non-local deployments.

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
