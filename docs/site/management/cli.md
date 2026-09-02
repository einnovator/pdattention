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
pra engine profiles local-vllm
pra engine storage local-vllm
pra engine sessions local-vllm
pra engine resources local-vllm
pra engine audit local-vllm
```

Use `--management-url URL` instead of a saved name for one-off automation. Most
read commands support `--json` and `--yaml`.

## Start a sidecar

```bash
pra engine serve \
  --engine sglang \
  --model Qwen/Qwen3-4B \
  --inference-url http://127.0.0.1:30000 \
  --port 9102 \
  --metrics-url http://127.0.0.1:9464/metrics \
  --trace-backend-url http://tempo:3200 \
  --grafana-url http://grafana:3000
```

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
