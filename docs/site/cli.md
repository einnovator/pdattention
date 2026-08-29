# PRA CLI and model onboarding

`pra` is the product command. `pra-hf` remains a deprecated alias for one
release cycle, and `pra-standalone` retains the from-scratch research trainer.

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

