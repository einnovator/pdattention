# Docker Compose

The default stack is deliberately profile-gated:

```bash
cd deploy/observability
docker compose --profile observability up -d
docker compose --profile observability ps
```

It starts an OTel Collector, Tempo, Prometheus, and Grafana. Plain
`docker compose up` does not start observability services.

## Multi-machine lab

Run one shared stack on a host reachable by every engine:

```bash
OBSERVABILITY_BIND_ADDRESS=0.0.0.0 \
GRAFANA_ADMIN_PASSWORD='choose-a-password' \
docker compose --profile observability up -d
```

List remote `/metrics` endpoints in
`prometheus-targets/engines.example.json`, or copy that example to a
deployment-specific JSON file in the same directory. Prometheus reloads target
files every 15 seconds. Point each engine or PRA wrapper at
`http://<collector-host>:4317` for OTLP/gRPC. Use
`engine_telemetry_probe.py` for engines that expose health and metrics but no
native OpenTelemetry exporter.

The LAN bind is intended for a trusted network. Use a reverse proxy, TLS, and
authentication before exposing Grafana, Prometheus, Tempo, or OTLP outside it.

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

For a single-host deployment Prometheus can reach a host endpoint through
`host.docker.internal`. In a federated deployment it scrapes the LAN targets
from the mounted discovery file, while every process exports spans to the
shared OTLP address.

## Stop and clean

```bash
docker compose --profile observability down
docker compose --profile observability down -v  # also removes local metrics data
```

The second command is destructive to observability volumes only.
