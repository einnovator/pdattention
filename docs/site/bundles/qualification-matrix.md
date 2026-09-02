# Bundle Qualification Matrix

Every row is scoped to the exact model revision, quantization, engine, profile, mode, hardware, and linked artifact. Family resemblance does not transfer qualification.

| Bundle | Engine | Recommended mode | Profile | Quality gate | Context saving | Evidence | Artifact |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| [`pra-qwen3-14b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-4bit) | mlx | Native Memory | BALANCED | 15/15 exact paired outputs | 89.1% | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/mac_scaling/qwen3_14b_mlx_profiles.json) |
| [`pra-qwen3-32b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-32b-mlx-4bit) | mlx | Native Memory | BALANCED | 15/15 exact paired outputs | 89.1% | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/mac_scaling/qwen3_32b_mlx_profiles.json) |
| [`pra-qwen3-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-4bit) | mlx | Native Memory | BALANCED | 15/15 exact paired outputs | 89.1% | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/mac_scaling/qwen3_8b_mlx_profiles.json) |
| [`pra-llama3-1-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-llama3-1-8b-mlx-4bit) | mlx | Selected Context | BALANCED | End-task pairing pending | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/llama3-8b/comparison.json) |
| [`pra-qwen3-4b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-4bit) | mlx | Selected Context | BALANCED | End-task pairing pending | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/qwen3-4b/comparison.json) |
| [`pra-gemma3-1b-mlx-4bit`](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-4bit) | mlx | Selected Context | BALANCED | End-task pairing pending | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/gemma3-1b/comparison.json) |
| [`pra-qwen3-0.6b`](https://huggingface.co/EInnovator/pra-qwen3-0.6b) | hf | Selected Context | BALANCED | Paired end-task qualification pending | NOT_MEASURED | RESEARCH | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/artifacts/pra_hf/bundles/pra-qwen3-0.6b/qualification/profile_evidence.json) |

## Evidence tiers

| Tier | Meaning |
| --- | --- |
| `PRODUCTION_QUALIFIED` | Production-scale workload, isolation, reliability, and economic gates passed. |
| `ENGINE_QUALIFIED` | Paired end-task and engine behavior measured for an exact identity. |
| `CONTROLLED` | Bounded controlled evidence; production generalization is not established. |
| `RESEARCH` | Mechanism research or dataset-specific component; not a deployment default. |
| `SMOKE` | Small feasibility check only. |
| `NOT_MEASURED` | The metric or condition has not been measured. |
| `NOT_APPLICABLE` | The metric does not apply to this realization. |
| `BLOCKED` | A known external or implementation dependency prevents measurement. |

Reduced consumer-layer profiles remain `CALIBRATION_PENDING`: held-out quality did not support promotion. BALANCED therefore retains all eligible consumer layers for the qualified MLX identities.
