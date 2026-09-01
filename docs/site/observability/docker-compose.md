# Docker Compose

The default stack is deliberately profile-gated:

```bash
cd deploy/observability
docker compose --profile observability up -d
docker compose --profile observability ps
```

It starts an OTel Collector, Prometheus, and Grafana. Plain `docker compose up`
does not start observability services.

## Container engine

Use both Compose files and select an engine profile:

```bash
MODEL=Qwen/Qwen3-0.6B \
docker compose -f docker-compose.yml -f docker-compose.engines.yml \
  --profile observability --profile vllm up -d
```

Examples are provided for vLLM, SGLang, OpenVINO Model Server,
TensorRT-LLM/Triton, llama.cpp, and Ollama. Model paths, image pins, accelerator
access, and credentials remain deployment inputs rather than checked-in
secrets.

## Hybrid host engine

MLX must remain on macOS/Metal. Embedded HF, AirLLM, and FreeToken may also be
more practical on the host. Start only the observability stack in containers,
then point the host process at it:

```bash
pra serve MODEL -e hf \
  --observability deploy/observability/observability.example.yaml
```

Prometheus reaches the host endpoint through `host.docker.internal`; Linux
Compose adds the host-gateway mapping. OTLP uses the loopback-published ports.

## Stop and clean

```bash
docker compose --profile observability down
docker compose --profile observability down -v  # also removes local metrics data
```

The second command is destructive to observability volumes only.
