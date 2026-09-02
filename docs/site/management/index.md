# Engine Management API

The **PRA Engine Management API** is the open-source control and observed-state
surface for one local PRA-enabled engine. It reports what that engine is doing,
which capabilities are actually available, and exposes a deliberately small set
of safe local operations.

It is not a fleet control plane. eInnovator Enterprise may aggregate and govern
many engines through this same public API, but using the REST API does not
require an enterprise service.

## Contract

| Property | Value |
| --- | --- |
| Protocol | `pra-management/1` |
| API prefix | `/v1/pra` |
| Default state | Disabled |
| Default bind | `127.0.0.1:9101` |
| Inference port | Separate |
| OpenAPI | `/openapi.json` |
| Swagger UI | `/docs` |
| ReDoc | `/redoc` |

The same contract is used by the HF reference runtime and sidecars for vLLM,
SGLang, MLX, OpenVINO, TensorRT-LLM, AirLLM, llama.cpp, Ollama, and FreeToken.
An adapter supplies capability and state providers plus only those actions its
engine can execute safely.

## Disabled by default

No listener, server thread, OpenAPI route, or authentication backend is created
unless the API is explicitly enabled. Start a standalone sidecar with:

```bash
pra engine serve --engine vllm --model Qwen/Qwen3-4B \
  --inference-url http://127.0.0.1:8000 --port 9101
```

The gateway has a separate management contract for upstream routing, transport,
and logical session state. Enable that API on its own default port with
`pra gateway serve --management-api --management-port 9150`; see
[Gateway Management API](gateway-rest-api.md). An upstream engine can expose
this engine API independently on `:9101`.

Unauthenticated mode is accepted only on a loopback bind. Production
deployments should use a private network, TLS, and bearer, OIDC, or mTLS
authentication.

## Registry registration

When a Registry URL is configured, the management runtime registers after its
listener starts, heartbeats in the background, and publishes changed observed
state. Inspect or retry it with `pra engine registry-status` and
`pra engine register`. `GET /v1/pra/registry` exposes only status and counters;
`POST /v1/pra/registry/register` performs an explicit retry. See
[Runtime Auto-Registration](../registry/runtime-auto-registration.md).

## Desired and observed state

The API is primarily an observed-state interface. `GET /v1/pra/config` reports
`desired_revision`, `observed_revision`, `in_sync`, and `drift_fields` without
pretending that a local engine owns fleet-wide desired state. Restart-required
topology changes are rejected with `409 RESTART_REQUIRED`.
