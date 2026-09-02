# Management CLI

The `pra engine` commands use the public REST API. They work with an attached
runtime listener or a separate engine sidecar.

## Connect and inspect

```bash
pra engine connect http://127.0.0.1:9101 --name local-vllm
pra engine health local-vllm
pra engine inspect local-vllm
```

Connections store a URL and optional token environment-variable name below
`PRA_HOME`; token values are never stored.

## Read state

```bash
pra engine capabilities local-vllm
pra engine config local-vllm
pra engine models local-vllm
pra engine model local-ollama qwen3:14b
pra engine profiles local-vllm
pra engine storage local-vllm
pra engine sessions local-vllm
pra engine resources local-vllm
pra engine audit local-vllm
```

Use `--management-url URL` instead of a saved name for one-off automation. Most
read commands support `--json` and `--yaml`.

## Dynamic model lifecycle

Only engines advertising the corresponding capabilities accept these commands:

```bash
pra engine load-model local-ollama qwen3:14b Qwen/Qwen3-14B \
  --bundle EInnovator/pra-qwen3-14b \
  --profile BALANCED --execution-mode selected-context
pra engine unload-model local-ollama qwen3:14b
```

The CLI checks `dynamic_model_load` or `dynamic_model_unload` before sending the
request. Ordinary vLLM, HF, SGLang, MLX, TensorRT-LLM, llama-server, AirLLM,
and FreeToken instances remain the simple one-model case.

## Start a sidecar

```bash
pra engine serve \
  --engine sglang \
  --model Qwen/Qwen3-4B \
  --inference-url http://127.0.0.1:30000 \
  --port 9102 \
  --metrics-url http://127.0.0.1:9464/metrics \
  --trace-backend-url http://tempo:3200 \
  --grafana-url http://grafana:3000 \
  --registry-instance-host 192.168.1.40 \
  --registry-management-url http://192.168.1.40:9102
```

For a genuine multi-model server, repeat `--model` and pair each entry with a
runtime alias:

```bash
pra engine serve --engine ollama \
  --model Qwen/Qwen3-14B --runtime-model-id qwen3:14b \
  --model google/gemma-3-12b-it --runtime-model-id gemma3:12b
```

When the listener binds to `0.0.0.0`, containers, WSL, and remote hosts may
derive a hostname that the Control Plane cannot resolve. Use the two Registry
address options to publish the route that fleet clients can actually reach.

A YAML file can hold listener policy:

```yaml
management_api:
  enabled: false
  host: 127.0.0.1
  port: 9101
  auth:
    mode: static_bearer
    token_env: PRA_MANAGEMENT_TOKEN
# For mode: mtls, also set tls_certfile, tls_keyfile, and tls_ca_certs.
```

`pra engine serve --config management.yaml` is itself explicit enablement; the
file default remains disabled so importing or installing PRA starts nothing.
