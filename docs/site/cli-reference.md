# CLI Command Reference

This page lists every public `pra` leaf command from the installed Click
command tree. Hidden compatibility and research controls are intentionally
excluded, exactly as they are from normal product help.

Every output block is representative and abridged. Paths, versions, measured
values, available devices, and recommendations depend on the local environment
and supplied evidence. Use `--json` or `--yaml` where offered for automation.

Use `pra COMMAND --help` as the runtime authority and this page for discoverable
examples. Start with the [CLI workflow guide](cli.md) for the qualification journey.

## Shared observability controls

Serving, Gateway, and Agent launch commands expose the same default-off controls:
`--observability`, `--otel`, `--otel-endpoint`, `--prometheus`, and
`--prometheus-port`. CLI overrides take precedence over the observability file
and conventional OTel environment variables. None auto-enable merely because a
collector or dashboard is present. See [Observability](observability.md).

## Gateway

### `pra gateway serve`

Serve logical PRA and OpenAI-compatible HTTP endpoints.

**Usage**

```text
pra gateway serve [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--config` | PATH | `-` | no | YAML gateway configuration. |
| `--host` | TEXT | `127.0.0.1` | no | Bind address for the local service. |
| `--port` | INTEGER | `8080` | no | TCP port for the local service. |
| `--mode` | TEXT | `passthrough` | no | Select the product execution or gateway mediation mode. |
| `--backend` | openai / sglang / freetoken / vllm / ollama / llama_cpp / mlx / custom / huggingface | `openai` | no | Select the downstream gateway adapter. |
| `--backend-url` | TEXT | `-` | no | Base URL of the existing downstream model endpoint. |
| `--model` | TEXT | `-` | no | Model identifier or local model path. |
| `-a`, `--pra-bundle` | TEXT | `auto` | no | Load a PRA bundle or configuration override. |
| `-p`, `--profile` | TEXT | `balanced` | no | Select a named PRA or agent profile. |
| `--prefix-cache-mode` | auto / unknown / stateless / automatic_prefix_cache / explicit_prefix_handle / session_state | `auto` | no | Declare or auto-detect downstream prefix-cache behavior. |
| `--session-state`, `--no-session-state` | flag | `-` | no | Enable or disable downstream session state. |
| `--incremental-messages`, `--full-messages` | flag | `-` | no | Send message deltas when supported, or full history. |
| `--resource-delta`, `--full-resources` | flag | `-` | no | Send resource operations when supported, or full inventories. |
| `--cache-affinity`, `--no-cache-affinity` | flag | `-` | no | Enable or disable stable cache-affinity hints. |
| `--fallback-injection` | before_current_user / system_suffix / tool_context / append_context_record / engine_native | `before_current_user` | no | Choose where Selected Context is inserted into ordinary messages. |
| `--sessions-dir` | PATH | `-` | no | Persist gateway session metadata under this directory. |
| `--observability` | PATH | `-` | no | Configure `observability`. |
| `--otel` | flag | `off` | no | Enable OpenTelemetry tracing explicitly. |
| `--otel-endpoint` | TEXT | `-` | no | Configure `otel-endpoint`. |
| `--prometheus` | flag | `off` | no | Enable the Prometheus endpoint explicitly. |
| `--prometheus-port` | INTEGER >= 1 <= 65535 | `-` | no | Configure `prometheus-port`. |
| `--management-api`, `--no-management-api` | flag | `-` | no | Explicitly enable the separate gateway management listener. |
| `--management-host` | TEXT | `-` | no | Management bind address; defaults to 127.0.0.1. |
| `--management-port` | INTEGER >= 1 <= 65535 | `-` | no | Management port; defaults to 9150. |
| `--management-auth-mode` | none / static_bearer / jwt_oidc / mtls | `-` | no | Management authentication; defaults to loopback-only no-auth. |
| `--management-token-env` | TEXT | `-` | no | Environment variable containing the management bearer token. |
| `--management-metrics-url` | TEXT | `-` | no | Configure `management-metrics-url`. |
| `--management-trace-url` | TEXT | `-` | no | Configure `management-trace-url`. |
| `--management-grafana-url` | TEXT | `-` | no | Configure `management-grafana-url`. |
| `--registry-url` | TEXT | `-` | no | Explicitly register this gateway with PRA Registry. |
| `--registry-token-env` | TEXT | `-` | no | Environment variable containing the Registry token; defaults to PRA_REGISTRY_TOKEN with --registry-url. |
| `--registry-instance-id` | TEXT | `-` | no | Stable Registry identity; otherwise it is persisted locally. |
| `--registry-instance-name` | TEXT | `-` | no | Human-readable managed gateway name. |
| `--registry-required` | flag | `off` | no | Fail startup when initial registration fails. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway serve --mode selected-context --backend vllm --backend-url http://127.0.0.1:8000/v1 --sessions-dir .pra/gateway-sessions
```

**Example output**

```text
PRA gateway on http://127.0.0.1:8080 -> vllm
Selected Context: enabled
Typed resource transport: disabled
Effective mode: Selected Context
```

### `pra gateway health`

Check gateway protocol and health.

**Usage**

```text
pra gateway health [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway health --management-url http://gateway:9150
```

**Example output**

```text
Status: healthy
Protocol: pra-gateway-management/1
Gateway Id: a31f...
```

### `pra gateway upstreams`

List configured upstream inference endpoints.

**Usage**

```text
pra gateway upstreams [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway upstreams --management-url http://gateway:9150 --json
```

**Example output**

```text
{"items": [{"upstream_id": "primary", "health": "healthy"}], "total": 1}
```

### `pra gateway sessions`

List privacy-safe gateway session summaries.

**Usage**

```text
pra gateway sessions [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway sessions --management-url http://gateway:9150 --json
```

**Example output**

```text
{"items": [{"session_id": "8e0f...", "known_resource_count": 3}], "total": 1}
```

### `pra gateway transport`

Show wire, delta, fallback, and reuse counters.

**Usage**

```text
pra gateway transport [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway transport --management-url http://gateway:9150 --yaml
```

**Example output**

```text
requested_mode: typed-transport
internal_transport: PRA-DELTA
wire_bytes: 18432
delta_bytes: 2048
```

### `pra gateway config`

Show effective gateway and policy configuration.

**Usage**

```text
pra gateway config [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway config --management-url http://gateway:9150 --yaml
```

**Example output**

```text
default_profile: BALANCED
policy:
  upstream_selection: failover
  session_affinity: true
```

### `pra gateway registry-status`

Show Registry registration and heartbeat state.

**Usage**

```text
pra gateway registry-status [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway registry-status --management-url http://gateway:9150
```

**Example output**

```text
enabled: true
status: online
instance_id: prod-gateway-01
heartbeat_success_total: 42
```

### `pra gateway register`

Retry this gateway's configured Registry registration immediately.

**Usage**

```text
pra gateway register [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway register --management-url http://gateway:9150
```

**Example output**

```text
instance_id: prod-gateway-01
instance_type: GATEWAY
status: ONLINE
```

### `pra gateway inspect`

Inspect gateway identity, capabilities, state, and observability.

**Usage**

```text
pra gateway inspect [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway inspect --management-url http://gateway:9150
```

**Example output**

```text
Info:
  Gateway Id: a31f...
Capabilities:
  Protocol: pra-gateway-management/1
State:
  Upstream Count: 2
```

### `pra gateway renegotiate`

Refresh one upstream capability handshake.

**Usage**

```text
pra gateway renegotiate [OPTIONS] UPSTREAM
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `UPSTREAM` | yes | Command input value. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--reason` | TEXT | `-` | yes | Record the operator reason in central audit. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway renegotiate primary --reason "refresh after engine restart" --management-url http://gateway:9150
```

**Example output**

```text
Action: renegotiate
Status: success
Target: primary
Native Memory: validated
```

### `pra gateway resync`

Invalidate one gateway session so its next turn fully resynchronizes.

**Usage**

```text
pra gateway resync [OPTIONS] SESSION
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SESSION` | yes | Command input value. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--reason` | TEXT | `-` | yes | Record the operator reason in central audit. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `--management-url` | TEXT | `http://127.0.0.1:9150` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Gateway management bearer token. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra gateway resync SESSION_HASH --reason "recover stale transport state" --management-url http://gateway:9150
```

**Example output**

```text
Action: resync-session
Status: success
Target: SESSION_HASH
```

## Engine management

### `pra engine connect`

Validate and remember a management URL without storing credentials.

**Usage**

```text
pra engine connect [OPTIONS] URL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `URL` | yes | PRA management API base URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name` | TEXT | `default` | no | Configure `name`. |
| `--token-env` | TEXT | `-` | no | Environment variable holding the bearer token; the secret is not stored. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine connect http://127.0.0.1:9101 --name local-vllm --token-env PRA_MANAGEMENT_TOKEN
```

**Example output**

```text
Name: local-vllm
URL: http://127.0.0.1:9101
Protocol: pra-management/1
Stored secret: false
```

### `pra engine health`

Check protocol and local engine health.

**Usage**

```text
pra engine health [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine health local-vllm
```

**Example output**

```text
Status: healthy
Protocol: pra-management/1
Instance ID: 2f9c...
```

### `pra engine config`

Show effective and desired configuration state.

**Usage**

```text
pra engine config [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine config local-vllm --yaml
```

**Example output**

```text
effective:
  profile: BALANCED
observed_revision: 1
in_sync: true
```

### `pra engine storage`

Show tier residency, quotas, and lifecycle counters.

**Usage**

```text
pra engine storage [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine storage local-vllm --json
```

**Example output**

```text
{
  "tiers": {"hot": {"bytes": 1048576, "resources": 2}},
  "maintenance_status": "running"
}
```

### `pra engine sessions`

List privacy-safe session summaries.

**Usage**

```text
pra engine sessions [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine sessions local-vllm --json
```

**Example output**

```text
{"items": [{"session_id": "8e0f...", "status": "active"}], "total": 1}
```

### `pra engine resources`

List privacy-safe resource summaries.

**Usage**

```text
pra engine resources [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine resources local-vllm --json
```

**Example output**

```text
{"items": [{"resource_id": "5a31...", "storage_tier": "hot"}], "total": 1}
```

### `pra engine models`

List loaded model identities.

**Usage**

```text
pra engine models [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine models local-vllm
```

**Example output**

```text
Items:
  Model Id: Qwen/Qwen3-4B
Total: 1
```

### `pra engine profiles`

List effective PRA profiles.

**Usage**

```text
pra engine profiles [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine profiles local-vllm
```

**Example output**

```text
Items:
  Name: BALANCED
  Qualification Status: VALIDATED
```

### `pra engine capabilities`

Show qualified local engine capabilities.

**Usage**

```text
pra engine capabilities [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine capabilities local-vllm
```

**Example output**

```text
Selected Context: AVAILABLE
Native Memory: CANDIDATE
Management API Version: pra-management/1
```

### `pra engine audit`

Show recent local management audit events.

**Usage**

```text
pra engine audit [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine audit local-vllm --json
```

**Example output**

```text
{"items": [{"event": "RESOURCE_PROMOTED", "result": "success"}]}
```

### `pra engine registry-status`

Show Registry registration and heartbeat state.

**Usage**

```text
pra engine registry-status [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine registry-status local-vllm --yaml
```

**Example output**

```text
enabled: true
status: online
instance_id: prod-vllm-01
heartbeat_success_total: 42
```

### `pra engine model`

Show one loaded model by its engine-local runtime identity.

**Usage**

```text
pra engine model [OPTIONS] TARGET RUNTIME_MODEL_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | yes | Saved connection name or management API URL. |
| `RUNTIME_MODEL_ID` | yes | Command input value. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine model local-ollama qwen3:14b --yaml
```

**Example output**

```text
Runtime Model Id: qwen3:14b
Model Id: Qwen/Qwen3-14B
Runtime State: loaded
```

### `pra engine load-model`

Load one model when the target engine supports dynamic residency.

**Usage**

```text
pra engine load-model [OPTIONS] TARGET RUNTIME_MODEL_ID MODEL_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | yes | Saved connection name or management API URL. |
| `RUNTIME_MODEL_ID` | yes | Command input value. |
| `MODEL_ID` | yes | Command input value. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `--bundle` | TEXT | `-` | no | Configure `bundle`. |
| `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `--execution-mode` | TEXT | `selected-context` | no | Configure `execution-mode`. |
| `--parameter` | TEXT; repeatable | `-` | no | Configure `parameters`. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine load-model local-ollama qwen3:14b Qwen/Qwen3-14B --profile balanced --execution-mode selected-context
```

**Example output**

```text
Action: load-model
Status: success
Runtime Model Id: qwen3:14b
```

### `pra engine unload-model`

Unload one runtime model and release its model-native state.

**Usage**

```text
pra engine unload-model [OPTIONS] TARGET RUNTIME_MODEL_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | yes | Saved connection name or management API URL. |
| `RUNTIME_MODEL_ID` | yes | Command input value. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--force` | flag | `off` | no | Allow engine-defined forced unload behavior. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine unload-model local-ollama qwen3:14b
```

**Example output**

```text
Action: unload-model
Status: success
Runtime Model Id: qwen3:14b
```

### `pra engine register`

Retry this engine's configured Registry registration immediately.

**Usage**

```text
pra engine register [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine register local-vllm --yaml
```

**Example output**

```text
instance_id: prod-vllm-01
instance_type: ENGINE
status: ONLINE
```

### `pra engine inspect`

Inspect engine identity, capabilities, state, and observability links.

**Usage**

```text
pra engine inspect [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine inspect --management-url http://127.0.0.1:9101
```

**Example output**

```text
Info:
  Engine: vllm
Capabilities:
  Management Api Version: pra-management/1
State:
  In Sync: true
```

### `pra engine patch-config`

Apply a bounded YAML/JSON configuration patch.

**Usage**

```text
pra engine patch-config [OPTIONS] [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--patch` | PATH | `-` | yes | Read a bounded YAML or JSON configuration patch. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine patch-config local-vllm --patch profile-patch.yaml
```

**Example output**

```text
Observed Revision: 2
In Sync: true
Effective:
  Profile: ECONOMY
```

### `pra engine action`

Run one bounded engine-supported local management action.

**Usage**

```text
pra engine action [OPTIONS] ACTION [TARGET]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `ACTION` | yes | Bounded management action name. |
| `TARGET` | no | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--management-url` | TEXT | `-` | no | Use a one-off PRA management API URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer its environment variable. |
| `--resource-id` | TEXT | `-` | no | Address a privacy-safe resource identifier returned by the API. |
| `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `--bundle` | TEXT | `-` | no | Configure `bundle`. |
| `--tenant-id` | TEXT | `-` | no | Authorize the storage action for this tenant. |
| `--idempotency-key` | TEXT | `-` | no | Deduplicate a retried management action. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--yaml` | flag | `off` | no | Emit machine-readable YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine action promote local-vllm --resource-id RESOURCE_ID --idempotency-key promote-42
```

**Example output**

```text
Action: promote
Status: success
Resource Id: RESOURCE_ID
Idempotent Replay: false
```

### `pra engine serve`

Start an explicitly enabled local management sidecar on a separate port.

**Usage**

```text
pra engine serve [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `--engine-version` | TEXT | `-` | no | Report the observed engine version when it cannot be discovered. |
| `--model` | TEXT; repeatable | `-` | no | Loaded model ID; repeat for multi-model engines. |
| `--runtime-model-id` | TEXT; repeatable | `-` | no | Engine-local model alias paired with each --model. |
| `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `--inference-url` | TEXT | `-` | no | Identify the engine inference endpoint managed by the sidecar. |
| `--config` | PATH | `-` | no | Configure `config-path`. |
| `--host` | TEXT | `-` | no | Bind address for the local service. |
| `--port` | INTEGER >= 1 <= 65535 | `-` | no | TCP port for the local service. |
| `--auth-mode` | none / static_bearer / jwt_oidc / mtls | `-` | no | Select local management authentication. |
| `--token-env` | TEXT | `-` | no | Name the environment variable containing a bearer token. |
| `--metrics-url` | TEXT | `-` | no | Publish the engine Prometheus endpoint link. |
| `--trace-backend-url` | TEXT | `-` | no | Publish the configured trace-backend link. |
| `--grafana-url` | TEXT | `-` | no | Publish the configured Grafana link. |
| `--tls-certfile` | PATH | `-` | no | Serve HTTPS with this certificate chain. |
| `--tls-keyfile` | PATH | `-` | no | Serve HTTPS with this private-key file. |
| `--tls-ca-certs` | PATH | `-` | no | Require client certificates issued by this CA bundle. |
| `--registry-url` | TEXT | `-` | no | Use this PRA Registry base URL. |
| `--registry-token-env` | TEXT | `PRA_REGISTRY_TOKEN` | no | Configure `registry-token-env`. |
| `--registry-instance-id` | TEXT | `-` | no | Configure `registry-instance-id`. |
| `--registry-instance-name` | TEXT | `-` | no | Configure `registry-instance-name`. |
| `--registry-instance-host` | TEXT | `-` | no | Externally reachable host advertised to the Registry. |
| `--registry-management-url` | TEXT | `-` | no | Externally reachable management API URL advertised to the Registry. |
| `--registry-required` | flag | `off` | no | Configure `registry-required`. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engine serve --engine vllm --model Qwen/Qwen3-4B --inference-url http://192.168.1.86:8000 --host 0.0.0.0 --port 9101 --registry-instance-host 192.168.1.86 --registry-management-url http://192.168.1.86:9101
```

**Example output**

```text
PRA management API (vllm) on http://127.0.0.1:9101
OpenAPI: http://127.0.0.1:9101/openapi.json
Swagger: http://127.0.0.1:9101/docs
```

## Registry

### `pra registry status`

Check registry protocol, database, and health.

**Usage**

```text
pra registry status [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry status --registry-url http://127.0.0.1:9200
```

**Example output**

```text
status: ok
protocol: pra-registry/1
database: postgresql
```

### `pra registry models`

List registered immutable model identities.

**Usage**

```text
pra registry models [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--offset` | INTEGER >= 0 | `0` | no | Skip this many registry records before returning results. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry models --limit 50 --json
```

**Example output**

```text
items:
- id: model-qwen3
  repo: Qwen/Qwen3-14B
  revision: a4d9b2d...
total: 1
```

### `pra registry bundles`

List PRA bundles and artifact references.

**Usage**

```text
pra registry bundles [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--offset` | INTEGER >= 0 | `0` | no | Skip this many registry records before returning results. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry bundles --limit 50 --json
```

**Example output**

```text
items:
- id: bundle-qwen3-14b
  immutable_revision: 9853a17...
  approval_state: APPROVED
total: 1
```

### `pra registry profiles`

List versioned PRA profiles.

**Usage**

```text
pra registry profiles [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--offset` | INTEGER >= 0 | `0` | no | Skip this many registry records before returning results. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry profiles --yaml
```

**Example output**

```text
items:
- id: balanced-qwen3
  version: '2'
  approval_state: APPROVED
total: 1
```

### `pra registry qualifications`

List immutable qualification evidence.

**Usage**

```text
pra registry qualifications [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--offset` | INTEGER >= 0 | `0` | no | Skip this many registry records before returning results. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry qualifications --limit 100 --json
```

**Example output**

```text
items:
- id: qasper-mlx-qwen3
  workload: qasper
  evidence_level: CONTROLLED
total: 1
```

### `pra registry deployments`

List desired deployment state.

**Usage**

```text
pra registry deployments [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--offset` | INTEGER >= 0 | `0` | no | Skip this many registry records before returning results. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry deployments --json
```

**Example output**

```text
items:
- id: production-mlx
  desired_revision: 7
  desired_mode: native-memory
total: 1
```

### `pra registry instances`

List self-registered engines and gateways with liveness filters.

**Usage**

```text
pra registry instances [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--type` | ENGINE / GATEWAY | `-` | no | Configure `instance-type`. |
| `--environment` | TEXT | `-` | no | Configure `environment`. |
| `--cluster` | TEXT | `-` | no | Configure `cluster`. |
| `--status` | ONLINE / DEGRADED / OFFLINE | `-` | no | Configure `status`. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--offset` | INTEGER >= 0 | `0` | no | Skip this many registry records before returning results. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry instances --type ENGINE --status ONLINE --json
```

**Example output**

```text
items:
- instance_id: prod-vllm-01
  instance_type: ENGINE
  status: ONLINE
  health: healthy
total: 1
```

### `pra registry instance`

Show one managed runtime and its observed/desired revisions.

**Usage**

```text
pra registry instance [OPTIONS] INSTANCE_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `INSTANCE_ID` | yes | Managed engine or gateway instance identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry instance prod-vllm-01 --yaml
```

**Example output**

```text
instance_id: prod-vllm-01
management_url: https://prod-vllm-01:9101
observed_revision: 4
in_sync: true
```

### `pra registry offline`

List runtimes whose heartbeat expired or which deregistered cleanly.

**Usage**

```text
pra registry offline [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--limit` | INTEGER >= 1 <= 500 | `50` | no | Maximum number of matching Hub bundles to return. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry offline --limit 100
```

**Example output**

```text
items:
- instance_id: old-engine-02
  status: OFFLINE
total: 1
```

### `pra registry resolve`

Resolve one model to a deterministic immutable PRA bundle.

**Usage**

```text
pra registry resolve [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--model-revision` | TEXT | `-` | no | Configure `model-revision`. |
| `--engine` | TEXT | `-` | no | Select the runtime or evidence-registry engine. |
| `--engine-version` | TEXT | `-` | no | Report the observed engine version when it cannot be discovered. |
| `--trust` | TEXT | `-` | no | Configure `trust`. |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry resolve Qwen/Qwen3-14B --engine vllm --trust einnovator-qualified
```

**Example output**

```text
selected_bundle:
  id: bundle-qwen3-14b
immutable_revision: 9853a17...
reason: highest approval, exact-revision, engine recommendation, then immutable identity
```

### `pra registry import-hf`

Import bundle metadata from an immutable Hugging Face revision.

**Usage**

```text
pra registry import-hf [OPTIONS] REPO_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `REPO_ID` | yes | Hugging Face repository identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry import-hf EInnovator/pra-qwen3-0.6b --revision 25e6907...
```

**Example output**

```text
model:
  repo: Qwen/Qwen3-0.6B
bundle:
  immutable_revision: 25e6907...
```

### `pra registry sync-hf-collection`

Import every PRA model bundle in a Hugging Face Collection.

**Usage**

```text
pra registry sync-hf-collection [OPTIONS] COLLECTION
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `COLLECTION` | yes | Hugging Face Collection slug to synchronize. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--registry-url` | TEXT | `-` | no | Registry base URL. |
| `--token` | TEXT | `-` | no | Bearer token; prefer the environment variable. |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry sync-hf-collection EInnovator/progressive-retrieval-attention
```

**Example output**

```text
collection: EInnovator/progressive-retrieval-attention
results:
- repo_id: EInnovator/pra-qwen3-0.6b
  status: IMPORTED
```

### `pra registry serve`

Start the standalone headless PRA Registry service.

**Usage**

```text
pra registry serve [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--host` | TEXT | `-` | no | Bind address for the local service. |
| `--port` | INTEGER >= 1 <= 65535 | `-` | no | TCP port for the local service. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra registry serve --config registry.yaml
```

**Example output**

```text
INFO: Uvicorn running on http://127.0.0.1:9200
Swagger: http://127.0.0.1:9200/docs
```

## Enterprise Control Plane

### `pra control serve`

Start the authenticated Control Plane backend and web application.

**Usage**

```text
pra control serve [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--host` | TEXT | `-` | no | Override the configured bind address. |
| `--port` | INTEGER >= 1 <= 65535 | `-` | no | Override the configured TCP port. |
| `--public-url` | TEXT | `-` | no | Browser-visible URL used for SSO callbacks. |
| `--reload` | flag | `off` | no | Reload the development server after source changes. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control serve --config control-plane.yaml --host 127.0.0.1 --port 9300
```

**Example output**

```text
INFO: Uvicorn running on http://127.0.0.1:9300
Control Plane: http://127.0.0.1:9300/index.html
```

### `pra control mcp`

Start the manager-backed MCP server over stdio or streamable HTTP.

**Usage**

```text
pra control mcp [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--transport` | stdio / http | `stdio` | no | Select MCP stdio or streamable HTTP transport. |
| `--host` | TEXT | `-` | no | Override the MCP HTTP bind address. |
| `--port` | INTEGER >= 1 <= 65535 | `-` | no | Override the MCP HTTP port. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control mcp --config control-plane.yaml --transport stdio
```

**Example output**

```text
PRA Control Manager MCP server started over stdio
```

### `pra control fleet`

List fleet state through the embedded manager.

**Usage**

```text
pra control fleet [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--auth-profile` | TEXT | `-` | no | Named service identity from control_plane.auth_profiles. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control fleet --config control-plane.yaml
```

**Example output**

```text
{
  "items": [{"name": "mlx-01", "status": "IN_SYNC"}],
  "summary": {"total": 1, "healthy": 1}
}
```

### `pra control inspect`

Inspect one engine without routing through REST.

**Usage**

```text
pra control inspect [OPTIONS] INSTANCE_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `INSTANCE_ID` | yes | Managed engine or gateway instance identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--section` | TEXT | `summary` | no | Select the engine state section to inspect. |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--auth-profile` | TEXT | `-` | no | Named service identity from control_plane.auth_profiles. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control inspect mlx-01 --section storage --config control-plane.yaml
```

**Example output**

```text
{
  "instance_id": "mlx-01",
  "section": "storage",
  "value": {"hot_bytes": 1048576}
}
```

### `pra control context`

Assemble deterministic task context through the manager.

**Usage**

```text
pra control context [OPTIONS] TASK
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `TASK` | yes | Task description used to assemble relevant context. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--repository` | TEXT | `-` | no | Associate task context with this repository. |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--auth-profile` | TEXT | `-` | no | Named service identity from control_plane.auth_profiles. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control context "work on MLX Native Memory" --repository einnovator/pdattention --config control-plane.yaml
```

**Example output**

```text
{
  "task": "work on MLX Native Memory",
  "repository": "einnovator/pdattention",
  "bundles": [],
  "limitations": []
}
```

### `pra control plan`

Create a durable action plan without applying it.

**Usage**

```text
pra control plan [OPTIONS] ACTION TARGET
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `ACTION` | yes | Bounded management action name. |
| `TARGET` | yes | Saved connection name or management API URL. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--values` | TEXT | `{}` | no | Requested change as a JSON object. |
| `--idempotency-key` | TEXT | `-` | no | Deduplicate a retried management action. |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--auth-profile` | TEXT | `-` | no | Named service identity from control_plane.auth_profiles. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control plan prefetch mlx-01 --values '{"resource_id":"document-42"}' --idempotency-key launch-42 --config control-plane.yaml
```

**Example output**

```text
{
  "plan_id": "...",
  "action": "prefetch",
  "impact": "low",
  "requires_confirmation": false
}
```

### `pra control apply`

Apply a durable plan using manager authorization and central audit.

**Usage**

```text
pra control apply [OPTIONS] PLAN_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `PLAN_ID` | yes | Durable action plan identifier returned by `pra control plan`. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--reason` | TEXT | `-` | yes | Record the operator reason in central audit. |
| `--confirm` | flag | `off` | no | Confirm a high-impact action plan. |
| `--idempotency-key` | TEXT | `-` | no | Deduplicate a retried management action. |
| `--config` | PATH | `-` | no | YAML Control Plane configuration. |
| `--auth-profile` | TEXT | `-` | no | Named service identity from control_plane.auth_profiles. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra control apply PLAN_ID --reason "prepare launch" --auth-profile operator --config control-plane.yaml
```

**Example output**

```text
{
  "plan_id": "...",
  "status": "applied",
  "idempotent_replay": false
}
```

## Environment and qualification

### `pra doctor`

Inspect the system, engines, local artifacts, and next action.

**Usage**

```text
pra doctor [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra doctor
```

**Example output**

```text
System: Python 3.10.x
Torch: AVAILABLE
Device backend: CPU
Next action: pra inspect MODEL --engine ENGINE
```

### `pra engines`

Show the registry-backed engine capability and recommendation matrix.

**Usage**

```text
pra engines [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--details` | TEXT | `-` | no | Show the detailed record for one engine. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra engines --details mlx
```

**Example output**

```text
Engine: MLX
Selected Context: available
Native Memory: measured
Recommended today: use the qualified profile for this model and hardware
```

### `pra inspect`

Inspect one MODEL and ENGINE as a deployable combination.

**Usage**

```text
pra inspect [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-a`, `--pra-bundle` | TEXT | `-` | no | Resolve and validate a bundle. Omit to discover published bundles without downloading them. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra inspect Qwen/Qwen3-0.6B --engine hf
pra inspect Qwen/Qwen3-0.6B --engine hf --pra-bundle auto
```

**Example output**

```text
Model: Qwen/Qwen3-0.6B
Revision: c1899de...
Engine: hf

Published PRA bundle found
  Repository: EInnovator/pra-qwen3-0.6b
  Revision: 25e6907...
  Base revision: c1899de...
  Compatibility: exact
  Trust: eInnovator-qualified

With --pra-bundle auto:
PRA bundle resolution
  Status: RESOLVED
  Compatibility: exact
```

### `pra evaluate`

Compare execution modes using one frozen selection and explicit gates.

**Usage**

```text
pra evaluate [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `-` | yes | Select the runtime or evidence-registry engine. |
| `-D`, `--dataset` | TEXT | `-` | yes | Name the evaluation dataset. |
| `--measurements` | PATH | `-` | no | Import measured mode results as JSON. |
| `--include-native-memory` | flag | `off` | no | Include Native Memory in the evaluation candidate set. |
| `--include-native-serving` | flag | `off` | no | Include Native Serving in the evaluation candidate set. |
| `--quality-threshold` | FLOAT >= 0.0 <= 1.0 | `0.95` | no | Minimum retained-quality ratio required by the gate. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-a`, `--pra-bundle` | TEXT | `auto` | no | Load a PRA bundle or configuration override. |
| `-p`, `--profile` | TEXT | `recommended` | no | Select a named PRA or agent profile. |
| `-o`, `--output` | PATH | `-` | no | Write artifacts to this file or directory. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra evaluate Qwen/Qwen3-1.7B --engine hf --dataset qasper --pra-bundle auto --measurements results.json -o .pra/runs/qasper
```

**Example output**

```text
Run: .pra/runs/qasper
Modes: full_context, selected_context
Measurements imported: results.json
Recommendation status: PENDING
```

### `pra recommend`

Recommend a mode from a completed qualification run.

**Usage**

```text
pra recommend [OPTIONS] RUN
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `RUN` | yes | Qualification or calibration run directory. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra recommend .pra/runs/qasper
```

**Example output**

```text
Recommended mode: selected_context
Reason: quality gate passed; native economics not qualified
```

### `pra report`

Export a qualification run as Markdown, HTML, or JSON.

**Usage**

```text
pra report [OPTIONS] RUN
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `RUN` | yes | Qualification or calibration run directory. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--format` | md / html / json | `md` | no | Choose the report output format. |
| `-o`, `--output` | PATH | `-` | no | Write artifacts to this file or directory. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra report .pra/runs/qasper --format html -o .pra/reports/qasper.html
```

**Example output**

```text
.pra/reports/qasper.html
```

### `pra qualify native-memory`

Compare Selected Context with frozen-selection HOT and WARM memory.

**Usage**

```text
pra qualify native-memory [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `-` | yes | Select the runtime or evidence-registry engine. |
| `-D`, `--dataset` | TEXT | `-` | yes | Name the evaluation dataset. |
| `--measurements` | PATH | `-` | no | Import measured mode results from JSON. |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `--quality-threshold` | FLOAT >= 0.0 <= 1.0 | `0.95` | no | Minimum retained-quality ratio required by the gate. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra qualify native-memory Qwen/Qwen3-1.7B --engine hf --dataset qasper --measurements results.json -o .pra/runs/native-memory
```

**Example output**

```text
Qualification: native_memory
Status: PASS
Run: .pra/runs/native-memory
```

### `pra qualify native-serving`

Measure scheduler-owned Native Serving beyond Native Memory.

**Usage**

```text
pra qualify native-serving [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `-` | yes | Select the runtime or evidence-registry engine. |
| `-D`, `--dataset` | TEXT | `-` | yes | Name the evaluation dataset. |
| `--measurements` | PATH | `-` | no | Import measured mode results from JSON. |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `--quality-threshold` | FLOAT >= 0.0 <= 1.0 | `0.95` | no | Minimum retained-quality ratio required by the gate. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra qualify native-serving Qwen/Qwen3-1.7B --engine vllm --dataset qasper --measurements results.json -o .pra/runs/native-serving
```

**Example output**

```text
Qualification: native_serving
Status: PENDING
Missing: concurrent scheduler measurements
```

## Assessments

### `pra assess init`

Create an editable assessment directory.

**Usage**

```text
pra assess init [OPTIONS] NAME
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | New assessment name. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--root` | PATH | `.pra/assessments` | no | Root directory for assessment workspaces. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra assess init customer-workload
```

**Example output**

```text
.pra/assessments/customer-workload/config.yaml
```

### `pra assess run`

Run the configured assessment and persist its evidence artifacts.

**Usage**

```text
pra assess run [OPTIONS] ASSESSMENT
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `ASSESSMENT` | yes | Assessment directory created by `pra assess init`. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--measurements` | PATH | `-` | no | Import measured mode results from JSON. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra assess run .pra/assessments/customer-workload --measurements results.json
```

**Example output**

```text
Assessment: customer-workload
Status: complete
Report data: .pra/assessments/customer-workload/run
```

### `pra assess report`

Regenerate an assessment report from its stored metrics.

**Usage**

```text
pra assess report [OPTIONS] ASSESSMENT
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `ASSESSMENT` | yes | Assessment directory created by `pra assess init`. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--format` | md / html / json | `md` | no | Choose the report output format. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra assess report .pra/assessments/customer-workload --format html
```

**Example output**

```text
.pra/assessments/customer-workload/report.html
```

## Models

### `pra model inspect`

Inspect MODEL without loading full weights by default.

**Usage**

```text
pra model inspect [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `--validate` | flag | `off` | no | Run structural validation during inspection. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra model inspect Qwen/Qwen3-1.7B --validate
```

**Example output**

```text
Family: qwen
Structural mapping: built-in
Validation requested: true
Status: candidate until validation completes
```

### `pra model adapt`

Generate a declarative structural adapter and validation record.

**Usage**

```text
pra model adapt [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-f`, `--force` | flag | `off` | no | Replace or rerun artifacts that already exist. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra model adapt Qwen/Qwen3-1.7B -o .pra/adapters/qwen3
```

**Example output**

```text
Adapter: .pra/adapters/qwen3/pra_adapter.yaml
Family: qwen
Learned weights: none
```

### `pra model validate`

Re-run the structural-adapter validation ladder.

**Usage**

```text
pra model validate [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-a`, `--adapter` | TEXT | `-` | no | Path or identifier of a structural adapter. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-s`, `--suite` | TEXT | `smoke` | no | Select the validation or calibration suite. |
| `-o`, `--output` | PATH | `-` | no | Write artifacts to this file or directory. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra model validate Qwen/Qwen3-1.7B --adapter .pra/adapters/qwen3 --suite standard -o .pra/runs/model-validation
```

**Example output**

```text
Suite: standard
Disabled-PRA parity: PASS
Native K/V capture: PASS
Generation: PASS
```

### `pra model onboard`

Run inspection, adaptation, validation, and runtime packaging.

**Usage**

```text
pra model onboard [OPTIONS] [MODEL]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | no | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--manifest` | PATH | `-` | no | Read a multi-model onboarding manifest. |
| `-s`, `--suite` | TEXT | `standard` | no | Select the validation or calibration suite. |
| `-o`, `--output` | PATH | `.pra/runs` | no | Write artifacts to this file or directory. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-f`, `--force` | flag | `off` | no | Replace or rerun artifacts that already exist. |
| `-j`, `--jobs` | INTEGER >= 1 | `1` | no | Maximum number of onboarding jobs. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra model onboard Qwen/Qwen3-1.7B --suite standard --engine hf -o .pra/runs/onboarding
```

**Example output**

```text
Runs: 1
Model: Qwen/Qwen3-1.7B
Output: .pra/runs/onboarding/Qwen--Qwen3-1.7B
```

## Learned adapters

### `pra adapter inspect`

Run this PRA operation.

**Usage**

```text
pra adapter inspect [OPTIONS] SOURCE
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE` | yes | Local artifact path or supported remote identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra adapter inspect .pra/adapters/router
```

**Example output**

```text
Adapter type: routing
Base model: Qwen/Qwen3-1.7B
Routing dimension: 128
```

### `pra adapter train routing`

Train a routing adapter under the dataset-level public namespace.

**Usage**

```text
pra adapter train routing [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-D`, `--dataset` | TEXT; repeatable | `-` | no | Record one or more training datasets; repeat the option. |
| `--validation` | TEXT; repeatable | `-` | no | Record one or more validation datasets; repeat the option. |
| `--train-features` | PATH; repeatable | `-` | no | Cached training feature file; repeat for multiple shards. |
| `--validation-features` | PATH; repeatable | `-` | no | Cached validation feature file; repeat for multiple shards. |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `--model-family` | qwen / llama / gemma3 | `-` | yes | Select the structural model-family mapping. |
| `--routing-dim` | INTEGER | `128` | no | Width of the learned routing projection. |
| `--steps` | INTEGER | `512` | no | Number of adapter optimization steps. |
| `--seed` | INTEGER | `53` | no | Random seed used by adapter training. |
| `-d`, `--device` | TEXT | `cuda` | no | Execution device such as auto, cpu, cuda, or mps. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra adapter train routing Qwen/Qwen3-1.7B --model-family qwen --train-features train.jsonl --validation-features valid.jsonl -D qasper -o .pra/adapters/router
```

**Example output**

```text
Output: .pra/adapters/router
Steps: 512
Validation metrics: .pra/adapters/router/metrics.json
```

### `pra adapter train memory`

Run this PRA operation.

**Usage**

```text
pra adapter train memory
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra adapter train memory
```

**Example output**

```text
Error: Memory-adapter training remains research-only; no certified dataset pipeline is packaged yet.
```

### `pra adapter train late-band`

Run this PRA operation.

**Usage**

```text
pra adapter train late-band
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra adapter train late-band
```

**Example output**

```text
Error: Late-band LoRA remains research-only and is not a certified product path.
```

### `pra adapter eval`

Run this PRA operation.

**Usage**

```text
pra adapter eval [OPTIONS] SOURCE
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE` | yes | Local artifact path or supported remote identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--features` | PATH; repeatable | `-` | yes | Feature file used for evaluation; repeat for multiple shards. |
| `--query-strategy` | TEXT | `last` | no | Choose how evaluation derives its routing query. |
| `-d`, `--device` | TEXT | `cuda` | no | Execution device such as auto, cpu, cuda, or mps. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra adapter eval .pra/adapters/router --features test.jsonl --query-strategy last
```

**Example output**

```text
Examples: 120
Top-k recall: 0.81
Mean reciprocal rank: 0.72
```

## Profiles

### `pra profiles show`

Run this PRA operation.

**Usage**

```text
pra profiles show [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-w`, `--workload` | TEXT | `-` | no | Filter or label profile evidence by workload. |
| `--registry` | PATH | `-` | no | Use an alternate profile benchmark registry. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra profiles show Qwen/Qwen3-1.7B --workload qasper
```

**Example output**

```text
Model: Qwen/Qwen3-1.7B
Workload: qasper
Profiles: REFERENCE_CORRECTNESS, BALANCED
```

### `pra profiles calibrate`

Run this PRA operation.

**Usage**

```text
pra profiles calibrate [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-s`, `--suite` | TEXT | `standard` | no | Select the validation or calibration suite. |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-w`, `--workload` | TEXT | `-` | no | Filter or label profile evidence by workload. |
| `--registry` | PATH | `-` | no | Use an alternate profile benchmark registry. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra profiles calibrate Qwen/Qwen3-1.7B --suite standard --engine hf --workload qasper -o .pra/runs/profile-calibration
```

**Example output**

```text
Output: .pra/runs/profile-calibration
Evidence tier: measured
Recommended profile: BALANCED
```

### `pra profiles compare`

Run this PRA operation.

**Usage**

```text
pra profiles compare [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-w`, `--workload` | TEXT | `-` | no | Filter or label profile evidence by workload. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra profiles compare Qwen/Qwen3-1.7B --workload qasper
```

**Example output**

```text
REFERENCE_CORRECTNESS: quality 1.000
BALANCED: quality 0.998
Reduced candidates: calibration pending
```

### `pra profiles report`

Run this PRA operation.

**Usage**

```text
pra profiles report [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra profiles report Qwen/Qwen3-1.7B -o .pra/reports/profiles.md
```

**Example output**

```text
.pra/reports/profiles.md
```

## Bundles

### `pra bundle build`

Run this PRA operation.

**Usage**

```text
pra bundle build [OPTIONS] RUN
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `RUN` | yes | Qualification or calibration run directory. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `--force` | flag | `off` | no | Replace a non-empty output directory. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra bundle build .pra/runs/profile-calibration -o .pra/bundles/qwen3
```

**Example output**

```text
Output: .pra/bundles/qwen3
Base model: Qwen/Qwen3-1.7B
Bundle schema: 2
```

### `pra bundle inspect`

Run this PRA operation.

**Usage**

```text
pra bundle inspect [OPTIONS] SOURCE
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE` | yes | Local artifact path or supported remote identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra bundle inspect .pra/bundles/qwen3
```

**Example output**

```text
Base model: Qwen/Qwen3-1.7B
Profiles: BALANCED
Evidence artifacts: present
```

### `pra bundle validate`

Run this PRA operation.

**Usage**

```text
pra bundle validate [OPTIONS] SOURCE
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE` | yes | Local artifact path or supported remote identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra bundle validate .pra/bundles/qwen3
```

**Example output**

```text
Status: VALID
Model: Qwen/Qwen3-1.7B
Schema version: 2
Checksums: verified
```

### `pra bundle card`

Generate or update a rich Hugging Face model card.

**Usage**

```text
pra bundle card [OPTIONS] SOURCE
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE` | yes | Local artifact path or supported remote identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--update` | flag | `off` | no | Write the generated card to README.md. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra bundle card .pra/bundles/qwen3 --update
```

**Example output**

```text
.pra/bundles/qwen3/README.md
```

### `pra bundle list`

List immutable bundles in the trusted auto-resolution registry.

**Usage**

```text
pra bundle list [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--model` | TEXT | `-` | no | Model identifier or local model path. |
| `--family` | TEXT | `-` | no | Filter trusted bundles by model family. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra bundle list --model Qwen/Qwen3-0.6B
```

**Example output**

```text
Bundles: 1
Qwen/Qwen3-0.6B -> owner/pra-qwen3-0.6b
Trust: eInnovator-qualified
```

### `pra bundle resolve`

Explain bundle selection and pin the resolved Hub revision.

**Usage**

```text
pra bundle resolve [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-a`, `--pra-bundle` | TEXT | `auto` | no | Load a PRA bundle or configuration override. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra bundle resolve Qwen/Qwen3-0.6B -e hf -a auto
```

**Example output**

```text
Status: RESOLVED
Revision: IMMUTABLE_COMMIT
Trust: eInnovator-qualified
Cache: HF snapshot cache
```

## Hugging Face Hub

### `pra hf login`

Run this PRA operation.

**Usage**

```text
pra hf login [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--check` | flag | `off` | no | Check existing Hub authentication without prompting. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf login --check
pra hf login
```

**Example output**

```text
Status: AUTHENTICATED
Name: maintainer
Organizations: EInnovator
```

### `pra hf list`

List pinned PRA bundles trusted for automatic resolution.

**Usage**

```text
pra hf list [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--query` | TEXT | `-` | no | Filter trusted metadata by a case-insensitive substring. |
| `--model` | TEXT | `-` | no | Require an exact base-model identifier. |
| `--family` | TEXT | `-` | no | Filter by model family or architecture. |
| `-e`, `--engine` | TEXT | `-` | no | Require compatibility with this engine. |
| `--qualification` | TEXT | `-` | no | Require an exact qualification tier. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf list --family qwen --engine mlx
```

**Example output**

```text
PRA bundle catalog (3)
Source: trusted-registry

EInnovator/pra-qwen3-0.6b
  Base model: Qwen/Qwen3-0.6B
  Qualification: CONTROLLED
  Trust: eInnovator-qualified
```

### `pra hf search`

Search live Hugging Face metadata for PRA model bundles.

**Usage**

```text
pra hf search [OPTIONS] [QUERY]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `QUERY` | no | Optional Hugging Face search text; defaults to `pra`. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--author` | TEXT | `EInnovator` | no | Limit results to one Hub namespace. |
| `--all-authors` | flag | `off` | no | Search all Hub namespaces; results remain untrusted unless registered. |
| `--limit` | INTEGER >= 1 <= 100 | `20` | no | Maximum number of matching Hub bundles to return. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf search qwen --author EInnovator --limit 20
```

**Example output**

```text
PRA bundle catalog (3)
Source: hugging-face-hub

EInnovator/pra-qwen3-0.6b
  Base model: Qwen/Qwen3-0.6B
  Qualification: CONTROLLED
  Trust: eInnovator-qualified
  Auto resolvable: True
```

### `pra hf pull`

Pull and validate a bundle, using the normal HF cache by default.

**Usage**

```text
pra hf pull [OPTIONS] REPO_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `REPO_ID` | yes | Hugging Face repository identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-o`, `--output` | PATH | `-` | no | Write artifacts to this file or directory. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf pull owner/pra-qwen3-0.6b --revision IMMUTABLE_COMMIT
```

**Example output**

```text
Repository: owner/pra-qwen3-0.6b
Resolved revision: IMMUTABLE_COMMIT
Cache path: HF snapshot cache
Status: VALID
```

### `pra hf push`

Run this PRA operation.

**Usage**

```text
pra hf push [OPTIONS] BUNDLE REPO_ID
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `BUNDLE` | yes | Local PRA bundle directory. |
| `REPO_ID` | yes | Hugging Face repository identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-y`, `--yes` | flag | `off` | no | Skip the interactive publication confirmation. |
| `--dry-run` | flag | `off` | no | Validate publication without uploading files. |
| `--private`, `--public` | flag | `off` | no | Set repository visibility when created. |
| `--collection` | TEXT | `-` | no | Collection slug, or namespace/name to create. |
| `--license` | TEXT | `-` | no | Assert a license only when it matches bundle provenance. |
| `--commit-message` | TEXT | `Publish PRA model bundle` | no | Set the Hugging Face upload commit message. |
| `--tag` | TEXT | `-` | no | Create an immutable release tag after upload. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf push .pra/bundles/qwen3 owner/pra-qwen3-0.6b --collection owner/pra-bundles --tag v0.2.0rc1 --dry-run
```

**Example output**

```text
Dry run: true
Repository: owner/Qwen3-PRA
Files checked: 8
Uploaded: 0
```

### `pra hf publish-manifest`

Validate or publish a resumable declarative bundle release list.

**Usage**

```text
pra hf publish-manifest [OPTIONS] MANIFEST
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MANIFEST` | yes | Command input value. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--dry-run` | flag | `off` | no | Validate publication without uploading files. |
| `-y`, `--yes` | flag | `off` | no | Skip the interactive publication confirmation. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf publish-manifest releases/pra_bundles.yaml --dry-run
```

**Example output**

```text
Manifest: releases/pra_bundles.yaml
Validated: 1
Uploaded: 0
Dry run: true
```

### `pra hf inspect`

Run this PRA operation.

**Usage**

```text
pra hf inspect [OPTIONS] SOURCE
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE` | yes | Local artifact path or supported remote identifier. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra hf inspect owner/Qwen3-PRA
```

**Example output**

```text
Source: owner/Qwen3-PRA
Base model: Qwen/Qwen3-1.7B
Bundle schema: 1
```

## Runtime and serving

### `pra runtime init`

Create a portable PRA runtime configuration directory.

**Usage**

```text
pra runtime init [OPTIONS] DIRECTORY
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `DIRECTORY` | yes | Runtime configuration directory to create. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--max-native-index-tokens` | INTEGER >= 1 | `-` | no | Set the native-index ingestion token budget. |
| `--max-native-index-bytes` | INTEGER >= 1 | `-` | no | Set the native-index ingestion byte budget. |
| `--defer-native-index` | flag | `off` | no | Build native selected-region state lazily. |
| `--storage` | memory / balanced / persistent / minimal | `balanced` | no | Select the semantic storage lifecycle profile. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra runtime init .pra/runtime --storage balanced --max-native-index-tokens 32768
```

**Example output**

```text
.pra/runtime/pra.yaml
Storage profile: balanced
Native index budget: 32768 tokens
```

### `pra runtime serve`

Serve MODEL with an explicit or policy-selected execution mode.

**Usage**

```text
pra runtime serve [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-m`, `--mode` | auto / selected-context / native-memory / native-serving | `auto` | no | Choose the qualified product execution mode. |
| `--explain` | flag | `off` | no | Explain mode evidence and resolution. |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-u`, `--endpoint` | TEXT | `-` | no | Use a remote runtime or gateway endpoint. |
| `--host` | TEXT | `127.0.0.1` | no | Bind address for the local service. |
| `--port` | INTEGER | `8000` | no | TCP port for the local service. |
| `-a`, `--pra-bundle` | TEXT | `-` | no | Load a PRA bundle or configuration override. |
| `-p`, `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `--storage` | memory / balanced / persistent / minimal | `balanced` | no | Select the semantic storage lifecycle profile. |
| `--storage-config` | PATH | `-` | no | Load a detailed storage policy file. |
| `--engine-arg` | TEXT; repeatable | `-` | no | Pass a provider-specific engine argument; repeat as needed. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--observability` | PATH | `-` | no | Configure `observability`. |
| `--otel` | flag | `off` | no | Configure `otel`. |
| `--otel-endpoint` | TEXT | `-` | no | Configure `otel-endpoint`. |
| `--prometheus` | flag | `off` | no | Configure `prometheus`. |
| `--prometheus-port` | INTEGER >= 1 <= 65535 | `-` | no | Configure `prometheus-port`. |
| `--management-api` | flag | `off` | no | Enable the separate PRA management listener. |
| `--management-host` | TEXT | `127.0.0.1` | no | Configure `management-host`. |
| `--management-port` | INTEGER >= 1 <= 65535 | `9101` | no | Configure `management-port`. |
| `--management-auth-mode` | none / static_bearer / jwt_oidc / mtls | `none` | no | Configure `management-auth-mode`. |
| `--management-token-env` | TEXT | `PRA_MANAGEMENT_TOKEN` | no | Configure `management-token-env`. |
| `--management-metrics-url` | TEXT | `-` | no | Configure `management-metrics-url`. |
| `--management-trace-url` | TEXT | `-` | no | Configure `management-trace-url`. |
| `--management-grafana-url` | TEXT | `-` | no | Configure `management-grafana-url`. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra runtime serve Qwen/Qwen3-1.7B --engine hf -m auto --explain --profile recommended --port 8000
```

**Example output**

```text
Runtime: hf
Status: healthy
Requested mode: auto
Resolved mode: selected-context
Reason: native economics require qualified evidence
Endpoint: http://127.0.0.1:8000
```

### `pra runtime inspect`

Run this PRA operation.

**Usage**

```text
pra runtime inspect [OPTIONS] [MODEL]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | no | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-u`, `--endpoint` | TEXT | `-` | no | Use a remote runtime or gateway endpoint. |
| `--host` | TEXT | `127.0.0.1` | no | Bind address for the local service. |
| `--port` | INTEGER | `8000` | no | TCP port for the local service. |
| `-a`, `--pra-bundle` | TEXT | `-` | no | Load a PRA bundle or configuration override. |
| `-p`, `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `--storage` | memory / balanced / persistent / minimal | `balanced` | no | Select the semantic storage lifecycle profile. |
| `--storage-config` | PATH | `-` | no | Load a detailed storage policy file. |
| `--engine-arg` | TEXT; repeatable | `-` | no | Pass a provider-specific engine argument; repeat as needed. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--observability` | PATH | `-` | no | Configure `observability`. |
| `--otel` | flag | `off` | no | Configure `otel`. |
| `--otel-endpoint` | TEXT | `-` | no | Configure `otel-endpoint`. |
| `--prometheus` | flag | `off` | no | Configure `prometheus`. |
| `--prometheus-port` | INTEGER >= 1 <= 65535 | `-` | no | Configure `prometheus-port`. |
| `--management-api` | flag | `off` | no | Enable the separate PRA management listener. |
| `--management-host` | TEXT | `127.0.0.1` | no | Configure `management-host`. |
| `--management-port` | INTEGER >= 1 <= 65535 | `9101` | no | Configure `management-port`. |
| `--management-auth-mode` | none / static_bearer / jwt_oidc / mtls | `none` | no | Configure `management-auth-mode`. |
| `--management-token-env` | TEXT | `PRA_MANAGEMENT_TOKEN` | no | Configure `management-token-env`. |
| `--management-metrics-url` | TEXT | `-` | no | Configure `management-metrics-url`. |
| `--management-trace-url` | TEXT | `-` | no | Configure `management-trace-url`. |
| `--management-grafana-url` | TEXT | `-` | no | Configure `management-grafana-url`. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra runtime inspect Qwen/Qwen3-1.7B --engine hf --storage balanced
```

**Example output**

```text
Engine: hf
Storage: balanced
Profile: provider default
Endpoint: embedded
```

### `pra runtime doctor`

Run this PRA operation.

**Usage**

```text
pra runtime doctor [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-u`, `--endpoint` | TEXT | `-` | no | Use a remote runtime or gateway endpoint. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra runtime doctor --engine hf
```

**Example output**

```text
Engine: hf
Provider: AVAILABLE
Model endpoint: not requested
Next action: pra runtime inspect MODEL --engine hf
```

### `pra runtime benchmark`

Run this PRA operation.

**Usage**

```text
pra runtime benchmark [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-p`, `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-o`, `--output` | PATH | `-` | yes | Write artifacts to this file or directory. |
| `--storage` | memory / balanced / persistent / minimal | `balanced` | no | Select the semantic storage lifecycle profile. |
| `--storage-config` | PATH | `-` | no | Load a detailed storage policy file. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra runtime benchmark Qwen/Qwen3-1.7B --engine hf --profile BALANCED --storage balanced -o .pra/benchmarks/qwen3
```

**Example output**

```text
Output: .pra/benchmarks/qwen3
Profile: BALANCED
Metrics: metrics.json
```

### `pra runtime capabilities`

Run this PRA operation.

**Usage**

```text
pra runtime capabilities [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra runtime capabilities --json
```

**Example output**

```text
{
  "typed_records": true,
  "native_memory": "engine-dependent",
  "streaming": true
}
```

### `pra serve`

Serve MODEL with an explicit or policy-selected execution mode.

**Usage**

```text
pra serve [OPTIONS] MODEL
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | Model identifier or local model path. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-m`, `--mode` | auto / selected-context / native-memory / native-serving | `auto` | no | Choose the qualified product execution mode. |
| `--explain` | flag | `off` | no | Explain mode evidence and resolution. |
| `-e`, `--engine` | TEXT | `hf` | no | Select the runtime or evidence-registry engine. |
| `-r`, `--revision` | TEXT | `-` | no | Pin a model, bundle, or Hub revision. |
| `-d`, `--device` | TEXT | `auto` | no | Execution device such as auto, cpu, cuda, or mps. |
| `-u`, `--endpoint` | TEXT | `-` | no | Use a remote runtime or gateway endpoint. |
| `--host` | TEXT | `127.0.0.1` | no | Bind address for the local service. |
| `--port` | INTEGER | `8000` | no | TCP port for the local service. |
| `-a`, `--pra-bundle` | TEXT | `-` | no | Load a PRA bundle or configuration override. |
| `-p`, `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `--storage` | memory / balanced / persistent / minimal | `balanced` | no | Select the semantic storage lifecycle profile. |
| `--storage-config` | PATH | `-` | no | Load a detailed storage policy file. |
| `--engine-arg` | TEXT; repeatable | `-` | no | Pass a provider-specific engine argument; repeat as needed. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--observability` | PATH | `-` | no | Configure `observability`. |
| `--otel` | flag | `off` | no | Configure `otel`. |
| `--otel-endpoint` | TEXT | `-` | no | Configure `otel-endpoint`. |
| `--prometheus` | flag | `off` | no | Configure `prometheus`. |
| `--prometheus-port` | INTEGER >= 1 <= 65535 | `-` | no | Configure `prometheus-port`. |
| `--management-api` | flag | `off` | no | Enable the separate PRA management listener. |
| `--management-host` | TEXT | `127.0.0.1` | no | Configure `management-host`. |
| `--management-port` | INTEGER >= 1 <= 65535 | `9101` | no | Configure `management-port`. |
| `--management-auth-mode` | none / static_bearer / jwt_oidc / mtls | `none` | no | Configure `management-auth-mode`. |
| `--management-token-env` | TEXT | `PRA_MANAGEMENT_TOKEN` | no | Configure `management-token-env`. |
| `--management-metrics-url` | TEXT | `-` | no | Configure `management-metrics-url`. |
| `--management-trace-url` | TEXT | `-` | no | Configure `management-trace-url`. |
| `--management-grafana-url` | TEXT | `-` | no | Configure `management-grafana-url`. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra serve Qwen/Qwen3-1.7B --engine hf -m auto --explain --profile recommended --port 8000
```

**Example output**

```text
Status: healthy
Requested mode: auto
Resolved mode: selected-context
Resolution reason: native economics require qualified evidence
```

## Agents

### `pra agent chat`

Open the persistent TUI; no flags uses the default profile.

**Usage**

```text
pra agent chat [OPTIONS] [LEGACY_MODEL]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `LEGACY_MODEL` | no | Optional compatibility spelling for the model; prefer `--model`. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--profile` | TEXT | `-` | no | Named agent profile. |
| `-P`, `--pra` | TEXT | `-` | no | PRA profile/bundle/config override. |
| `-m`, `--model` | TEXT | `-` | no | Model identifier or local model path. |
| `-e`, `--engine` | TEXT | `-` | no | Select the runtime or evidence-registry engine. |
| `-u`, `--endpoint` | TEXT | `-` | no | Use a remote runtime or gateway endpoint. |
| `-c`, `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `-w`, `--workspace` | PATH | `-` | no | Set the agent workspace directory. |
| `-s`, `--skills` | PATH; repeatable | `-` | no | Discover skills under this directory; repeat as needed. |
| `--context-transport` | auto / pra / text | `-` | no | Require typed PRA, require text, or negotiate automatically. |
| `--allow-text-fallback`, `--no-text-fallback` | flag | `-` | no | Allow or reject explicit Selected Context fallback. |
| `--session` | TEXT | `-` | no | Use this agent session identifier. |
| `-r`, `--resume` | flag | `off` | no | Resume persisted session state. |
| `-t`, `--task` | TEXT | `-` | no | Set or update the active task description. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--observability` | PATH | `-` | no | Configure `observability`. |
| `--otel` | flag | `off` | no | Enable OpenTelemetry tracing explicitly. |
| `--otel-endpoint` | TEXT | `-` | no | Configure `otel-endpoint`. |
| `--prometheus` | flag | `off` | no | Enable Prometheus metrics explicitly. |
| `--prometheus-port` | INTEGER >= 1 <= 65535 | `-` | no | Configure `prometheus-port`. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra agent chat --model Qwen/Qwen3-0.6B --workspace . --task "Inspect this repository"
```

**Example output**

```text
Agent profile: default
Runtime: embedded/hf
Session: new
> Inspect this repository
```

### `pra agent run`

Run one noninteractive turn from an argument or stdin.

**Usage**

```text
pra agent run [OPTIONS] [PROMPT]
```

**Arguments**

| Argument | Required | Description |
| --- | --- | --- |
| `PROMPT` | no | One noninteractive agent instruction; stdin is used when omitted. |

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--profile` | TEXT | `-` | no | Named agent profile. |
| `-P`, `--pra` | TEXT | `-` | no | PRA profile/bundle/config override. |
| `-m`, `--model` | TEXT | `-` | no | Model identifier or local model path. |
| `-e`, `--engine` | TEXT | `-` | no | Select the runtime or evidence-registry engine. |
| `-u`, `--endpoint` | TEXT | `-` | no | Use a remote runtime or gateway endpoint. |
| `-c`, `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `-w`, `--workspace` | PATH | `-` | no | Set the agent workspace directory. |
| `-s`, `--skills` | PATH; repeatable | `-` | no | Discover skills under this directory; repeat as needed. |
| `--context-transport` | auto / pra / text | `-` | no | Require typed PRA, require text, or negotiate automatically. |
| `--allow-text-fallback`, `--no-text-fallback` | flag | `-` | no | Allow or reject explicit Selected Context fallback. |
| `--session` | TEXT | `-` | no | Use this agent session identifier. |
| `-r`, `--resume` | flag | `off` | no | Resume persisted session state. |
| `-t`, `--task` | TEXT | `-` | no | Set or update the active task description. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--json` | flag | `off` | no | Emit machine-readable JSON. |
| `--observability` | PATH | `-` | no | Configure `observability`. |
| `--otel` | flag | `off` | no | Enable OpenTelemetry tracing explicitly. |
| `--otel-endpoint` | TEXT | `-` | no | Configure `otel-endpoint`. |
| `--prometheus` | flag | `off` | no | Enable Prometheus metrics explicitly. |
| `--prometheus-port` | INTEGER >= 1 <= 65535 | `-` | no | Configure `prometheus-port`. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra agent run --profile work --session issue-42 "Summarize the current task state." --json
```

**Example output**

```text
{
  "response": "The current task is ...",
  "session_id": "issue-42",
  "tool_calls": 0
}
```

### `pra agent inspect`

Run this PRA operation.

**Usage**

```text
pra agent inspect [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `-c`, `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra agent inspect --profile work --yaml
```

**Example output**

```text
agent_profile: work
model: Qwen/Qwen3-0.6B
context_transport: auto
tools: ask
```

### `pra agent start`

Start the experimental optional FastAPI agent UI.

**Usage**

```text
pra agent start [OPTIONS]
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--profile` | TEXT | `-` | no | Select a named PRA or agent profile. |
| `-P`, `--pra` | TEXT | `-` | no | Override the PRA profile, bundle, or configuration for the agent. |
| `-c`, `--config` | PATH | `-` | no | Load an explicit agent profile document. |
| `-h`, `--host` | TEXT | `127.0.0.1` | no | Bind address for the local service. |
| `--port` | INTEGER | `8765` | no | TCP port for the local service. |
| `-d`, `--detach` | flag | `off` | no | Run the Web UI as a detached process. |
| `-o`, `--open` | flag | `off` | no | Open the Web UI in the default browser. |
| `-v`, `--verbose` | flag | `off` | no | Show additional resolution and diagnostic detail. |
| `--json` | flag | `off` | no | Emit JSON. |
| `--yaml` | flag | `off` | no | Emit YAML. |
| `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra agent start --profile work --host 127.0.0.1 --port 8765 --detach --open
```

**Example output**

```text
PRA Agent Web UI started
URL: http://127.0.0.1:8765
Detached: true
```

### `pra agent stop`

Safely stop a detached PRA Agent Web UI.

**Usage**

```text
pra agent stop
```

**Options**

| Option | Value | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `-h`, `--help` | flag | `off` | no | Show command help and exit. |

**Common use**

```bash
pra agent stop
```

**Example output**

```text
PRA Agent Web UI stopped.
```

## Exit behavior

Successful commands return exit status `0`. Usage errors, unavailable optional
dependencies, rejected capability requirements, and failed validation return a
nonzero status. Server commands remain attached unless their command explicitly
supports detaching.

_Generated from `pra_hf.cli`; do not edit this page manually._
