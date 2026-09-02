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
