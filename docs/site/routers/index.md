# PRA routers

A PRA router exposes stable logical names such as `pra/qwen3-32b` while the PRA Registry decides which deployments are eligible. The routing data plane handles request transport, retries, fallback, streaming, and fast replica selection.

```text
Application -> routing data plane -> PRA Gateway or engine

PRA Control Plane -> PRA Registry -> PRA Router Controller -> router configuration
```

The Registry and Control Plane never proxy inference traffic. If either control service is unavailable, the router continues serving its last-good configuration.

## Routing levels

| Level | Owner | Decision |
|---|---|---|
| Logical model | PRA policy | Public model or family |
| Deployment or pool | PRA Registry and Router Controller | Eligible qualified engine pool |
| Replica or pod | Router, endpoint picker, or engine scheduler | Concrete request target |

PRA initially uses deterministic policy. It does not train a learned fleet router.

## Supported data planes

| Router | Best fit | Configuration path |
|---|---|---|
| PRA reference router | Labs and 1-20 engines | Native management API |
| LiteLLM | OpenAI-compatible multi-provider gateway | Generated model groups |
| agentgateway | LLM, MCP, and A2A regional traffic | Watched file; xDS-ready model |
| Kubernetes GAIE / llm-d | Elastic Kubernetes fleets | `HTTPRoute` and `InferencePool` |
| Bifrost | Dynamic providers, governance rules, fallback | Provider and routing-rule model |

## Quick start

```bash
pip install 'pra-hf[router]'
pra router serve --config reference-router.yaml
pra router routes
```

Use `pra router controller --once` to reconcile every router registered in a local Registry.

## Control-path scale smoke test

The reproducible offline benchmark compiles the same one-route desired state at
10, 100, and 1,000 eligible backends. Results below are mean compile latency on
the Windows development host; they do **not** measure router apply convergence
or request throughput.

| Adapter | 10 backends | 100 backends | 1,000 backends | 1,000-backend config |
|---|---:|---:|---:|---:|
| LiteLLM | 0.12 ms | 0.99 ms | 11.00 ms | 530,820 B |
| agentgateway | 0.09 ms | 0.82 ms | 11.97 ms | 167,854 B |
| Kubernetes GAIE | 0.02 ms | 0.01 ms | 0.01 ms | 769 B |
| PRA reference | 0.03 ms | 0.25 ms | 5.00 ms | 499,983 B |
| Bifrost | 0.10 ms | 1.14 ms | 14.12 ms | 454,880 B |

The GAIE output remains constant because PRA emits pool selectors and delegates
pod membership to Kubernetes discovery. Run the benchmark again with:

```bash
python -m experiments.paper4_5_runtime.benchmark_router_control
```

The raw result is stored in
`docs/papers/shared/results/paper4_5_runtime/router_control_scale.json`.
