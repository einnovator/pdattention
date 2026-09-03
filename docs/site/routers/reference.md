# PRA reference router

The reference router is a small async Python data plane for local labs, Windows/macOS/Linux demonstrations, and integration tests. It is designed for roughly 1-20 engines, not as the preferred thousand-replica enterprise router.

## Configuration

```yaml
revision: 1
max_attempts: 2
routes:
  - id: qwen32
    public_model: pra/qwen3-32b
    strategy: least-active
    backends:
      - id: mlx-m5
        url: http://m5.example:8080
        model: qwen3-32b
        weight: 2
        region: eu
      - id: vllm-cuda
        url: http://cuda.example:8000
        model: qwen3-32b
        weight: 1
        region: eu
```

```bash
pra router serve --config reference-router.yaml --host 127.0.0.1 --port 9400
```

Point any OpenAI client at `http://127.0.0.1:9400/v1` and request `model="pra/qwen3-32b"`.

## Policies

Supported policies are `round-robin`, `random`, `weighted`, `least-active`, `least-busy`, and `lowest-recent-ttft`. Retry occurs only before a streaming response emits bytes. A partial stream is never replayed into another backend.

## Management

| Endpoint | Purpose |
|---|---|
| `/health` | Liveness and active revision |
| `/v1/router/info` | Routes, backends, activity, and scope |
| `/v1/router/routes` | Logical routes |
| `/v1/router/backends` | Configured endpoints |
| `/v1/router/metrics` | Requests, errors, retries, latency, and active requests |
| `/v1/router/config` | Controller read/apply path |

Only `traceparent`, `tracestate`, `baggage`, and documented `x-pra-*` request metadata are forwarded. Authorization headers and Registry metadata are not copied downstream.
