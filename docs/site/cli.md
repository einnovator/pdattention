# PRA CLI and model onboarding

`pra` is the product command. `pra-hf` remains a deprecated alias for one
release cycle, and `pra-standalone` retains the from-scratch research trainer.

See the generated [CLI Command Reference](cli-reference.md) for every public
command, complete option tables, common-use examples, and representative output.

## Qualification journey

The shortest product workflow is:

```bash
pra doctor
pra engines
pra inspect Qwen/Qwen3-1.7B --engine hf
pra evaluate Qwen/Qwen3-1.7B --engine hf --dataset qasper \
  --measurements matched-results.json -o .pra/runs/qwen-qasper
pra recommend .pra/runs/qwen-qasper
pra report .pra/runs/qwen-qasper --format html
pra serve Qwen/Qwen3-1.7B --engine hf --profile recommended --mode auto
```

`doctor` groups system, engine, local-artifact, problem, and next-action
information. `engines` reads the same versioned registry that generates this
site; use `pra engines --details mlx` for provenance and limitations.

`inspect` joins model metadata with an engine's product capabilities. It does
not infer qualification from the existence of a code path.

## Evaluate and recommend

`pra evaluate` always creates a Full Context versus Selected Context record.
Add `--include-native-memory` or `--include-native-serving` only when those
paths should enter the qualification. The command accepts a mode-measurement
JSON document and writes a reproducible run directory:

```text
config.yaml
environment.json
quality.json
metrics.json
runs/
report.md
recommendation.json
```

The input JSON uses product mode names under `modes`:

```json
{
  "selector_digest": "sha256-of-frozen-selection",
  "hardware": "deployment hardware identity",
  "modes": {
    "full_context": {
      "quality": {"f1": 0.81},
      "context": {"visible_input_tokens": 8192},
      "performance": {"ttft_p95_ms": 640.0}
    },
    "selected_context": {
      "quality": {"f1": 0.80},
      "context": {"visible_input_tokens": 2200},
      "performance": {"ttft_p95_ms": 270.0}
    }
  }
}
```

Missing values remain JSON `null` and render as `NOT_MEASURED`. An evaluation
without imported measurements is a valid assessment template, not a synthetic
benchmark. It receives no production recommendation until its quality gate
passes.

Attribution is adjacent and explicit:

- Full Context to Selected Context measures retrieval and visible-token gains.
- Selected Context to Native Memory measures representation and residency.
- Native Memory to Native Serving measures scheduler ownership.

`pra recommend` cannot promote Native Memory merely because native execution is
implemented. It requires frozen selection, semantic parity, HOT/WARM latency,
memory, reuse, reference-encoding cost, qualified engine evidence, and a
positive incremental economic result. Negative AirLLM evidence keeps Selected
Context as the recommendation.

Export with `pra report RUN --format md|html|json`.

## Enterprise assessment

The assessment wrapper makes the same artifacts easier to hand to a design
partner:

```bash
pra assess init customer-workload
# Edit .pra/assessments/customer-workload/config.yaml
pra assess run .pra/assessments/customer-workload --measurements matched-results.json
pra assess report .pra/assessments/customer-workload --format html
```

The command runs inside the Python environment where it was installed.
Accordingly, `pra doctor` reports that environment's Torch build and device
backends. A CPU-only virtual environment reports CPU-only Torch even when a
different Python installation on the same host has CUDA. `MPS` denotes the
PyTorch Apple Metal backend; it is a device backend rather than an engine.

```bash
pra doctor
pra model inspect Qwen/Qwen3-1.7B
pra model adapt Qwen/Qwen3-1.7B -o .pra/adapters/qwen3
pra model validate Qwen/Qwen3-1.7B -a .pra/adapters/qwen3
pra profiles calibrate Qwen/Qwen3-1.7B -s standard -o .pra/runs/qwen3
pra bundle build .pra/runs/qwen3 -o .pra/bundles/qwen3
pra hf push .pra/bundles/qwen3 owner/Qwen3-PRA --dry-run
```

The command groups keep four concepts separate:

| Surface | Responsibility |
| --- | --- |
| `model` | architecture inspection and training-free structural adapters |
| `adapter` | learned routing, memory-use, or late-band adapters |
| `profiles` | measured semantic and physical calibration |
| `hf` | Hub authentication and artifact transport only |

The gateway accepts product-facing deployment names. Normal help and output do
not expose research protocol labels:

```bash
pra gateway serve --mode passthrough --backend openai --backend-url URL
pra gateway serve --mode selected-context --backend vllm --backend-url URL
pra gateway serve --mode typed-transport --backend sglang --backend-url URL
```

`selected-context` renders authorized selected records for an ordinary engine.
`typed-transport` preserves typed resources for a capable endpoint. Legacy
research spellings remain accepted for reproduction but are not used in public
deployment examples.

## Structural adapters

A structural adapter maps a conventional decoder's layers, attention module,
Q/K/V/O projections, head geometry, masking, and positional encoding. It does
not contain learned weights. Conventional Qwen, Llama, and Gemma mappings can be
written as versioned `pra_adapter.yaml`; unusual architectures require a
reviewed Python plugin.

`pra model validate` records V0 through V9 separately. Projection discovery
passes only V0. Weight loading, disabled-PRA parity, native K/V capture,
visible-prefix equivalence, source-relative RoPE, cached decode, GQA/MQA,
selected-region consumption, and generation remain explicit gates.

## Profiles and bundles

Calibration writes `pra.yaml`, `profiles.yaml`, `metrics.json`,
`benchmarks.json`, `environment.json`, `manifest.json`, and `report.md`.
Unvalidated optima remain `QUALITY_MAX_CANDIDATE` with
`SMOKE / CALIBRATION_PENDING`; they are not silently promoted to `QUALITY_MAX`.

`PRAModelBundle` packages base-model identity, structural and learned adapters,
profiles, benchmark evidence, runtime compatibility, engine realizations, and
provenance. A bundle does not duplicate base-model weights. Local directories,
Hub repository IDs, and pinned Hub revisions are accepted by the shared bundle
resolver when the optional Hub dependency is installed.

Human-readable output is the default. Add `--json` or `--yaml` for scripts.
Product configuration precedence is explicit CLI values, command YAML, bundle
or profile, project config, user config, then package defaults.

## Runtime modes

Both `pra serve` and `pra runtime serve` accept:

- `--mode selected-context` for the portable baseline;
- `--mode native-memory` for an explicitly qualified native path;
- `--mode auto` for conservative policy selection.

`auto` does not promote a deeper mode from capability availability alone. It
reports the requested mode, selected mode, and reason. `--profile recommended`
resolves to the current conservative `BALANCED` profile; reduced consumer-layer
profiles remain calibration candidates until workload evidence qualifies them.
# Storage profiles

The runtime commands accept a named semantic storage profile or a detailed
YAML policy. Inspection emits the fully resolved policy.

```powershell
pra runtime serve Qwen/Qwen3-1.7B -e hf --storage balanced
pra runtime inspect Qwen/Qwen3-1.7B -e mlx --storage-config storage.yaml --json
pra runtime benchmark Qwen/Qwen3-1.7B -e hf --storage minimal -o .pra/bench
```

See [Semantic storage lifecycle](storage.md) for tier and retention semantics.
