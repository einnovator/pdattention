# Bundle Qualification Matrix

Every row is scoped to the exact model revision, quantization, engine, profile, mode, hardware, and linked artifact. Family resemblance does not transfer qualification.

| Bundle | Engine | Recommended mode | Profile | Quality gate | Context saving | Evidence | Artifact |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| [`pra-qwen3-14b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-4bit) | mlx | Native Memory | BALANCED | 15/15 exact paired outputs | 89.1% | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/mac_scaling/qwen3_14b_mlx_profiles.json) |
| [`pra-qwen3-32b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-32b-mlx-4bit) | mlx | Native Memory | BALANCED | 15/15 exact paired outputs | 89.1% | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/mac_scaling/qwen3_32b_mlx_profiles.json) |
| [`pra-qwen3-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-4bit) | mlx | Native Memory | BALANCED | 15/15 exact paired outputs | 89.1% | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/mac_scaling/qwen3_8b_mlx_profiles.json) |
| [`pra-qwen2-5-1-5b-instruct`](https://huggingface.co/EInnovator/pra-qwen2-5-1-5b-instruct) | hf | Selected Context | BALANCED | Learned QASPER R@20% +0.085; combined +0.017 | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/qwen2.5-1.5b-instruct/comparison.json) |
| [`pra-qwen2-5-coder-1-5b-instruct`](https://huggingface.co/EInnovator/pra-qwen2-5-coder-1-5b-instruct) | hf | Selected Context | BALANCED | Learned QASPER R@20% +0.221; combined +0.019 | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/qwen2.5-coder-1.5b-instruct/comparison.json) |
| [`pra-llama3-1-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-llama3-1-8b-mlx-4bit) | mlx | Selected Context | BALANCED | End-task pairing pending | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/llama3-8b/comparison.json) |
| [`pra-qwen3-4b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-4bit) | mlx | Selected Context | BALANCED | End-task pairing pending | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/qwen3-4b/comparison.json) |
| [`pra-gemma3-1b-mlx-4bit`](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-4bit) | mlx | Selected Context | BALANCED | End-task pairing pending | NOT_MEASURED | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters/gemma3-1b/comparison.json) |
| [`pra-qwen3-0.6b`](https://huggingface.co/EInnovator/pra-qwen3-0.6b) | hf | Selected Context | BALANCED | Paired end-task qualification pending | NOT_MEASURED | RESEARCH | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/artifacts/pra_hf/bundles/pra-qwen3-0.6b/qualification/profile_evidence.json) |
| [`pra-qwen3-4b-mlx-8bit`](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-8bit) | mlx | Native Memory | BALANCED | 60/60 exact paired outputs; learned QASPER R@20% +0.120 but HotpotQA -0.227 | 91.5% fewer visible tokens | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/qwen3-4b-mlx-8bit/qualification_summary.json) |
| [`pra-qwen3-8b-mlx-8bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-8bit) | mlx | Native Memory | BALANCED | 60/60 exact paired outputs across QASPER, HotpotQA, and 2Wiki | 91.5% fewer visible tokens | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/qwen3-8b-mlx-8bit/qualification_summary.json) |
| [`pra-qwen3-14b-mlx-8bit`](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-8bit) | mlx | Native Memory | BALANCED | 60/60 exact paired outputs across QASPER, HotpotQA, and 2Wiki | 91.5% fewer visible tokens | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/qwen3-14b-mlx-8bit/qualification_summary.json) |
| [`pra-qwen3-8b-mlx-6bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-6bit) | mlx | Native Memory | BALANCED | 60/60 exact paired outputs across QASPER, HotpotQA, and 2Wiki | 91.5% fewer visible tokens | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/qwen3-8b-mlx-6bit/qualification_summary.json) |
| [`pra-llama3-2-1b-mlx-8bit`](https://huggingface.co/EInnovator/pra-llama3-2-1b-mlx-8bit) | mlx | Native Memory | BALANCED | 60/60 exact paired outputs across QASPER, HotpotQA, and 2Wiki | 91.5% fewer visible tokens | ENGINE_QUALIFIED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/llama3.2-1b-mlx-8bit/qualification_summary.json) |
| [`pra-gemma3-1b-mlx-8bit`](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-8bit) | mlx | Selected Context | BALANCED | 3/60 exact paired outputs; mean token-F1 delta -0.0036, so Native Memory remains candidate-only | 91.1% fewer visible tokens | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/gemma3-1b-mlx-8bit/qualification_summary.json) |
| [`pra-qwen2-5-1-5b-instruct-bnb-8bit`](https://huggingface.co/EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit) | hf | Selected Context | BALANCED | 0/15 exact paired outputs; mean token-F1 delta -0.3813, so Native Memory remains candidate-only | 91.7% fewer visible tokens | CONTROLLED | [source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/docs/papers/shared/results/paper4_5_runtime/exact_identity_qualification/qwen2.5-1.5b-instruct-bnb-8bit/qualification_summary.json) |

## Canonical condition audit

This audit asks whether the same task, exact model, engine/hardware, mode, and profile have been measured under all three conditions. `AVAILABLE_EXISTING` here means that at least the quality, context, serving, and memory fields present in the linked selector-frozen artifact can be imported; it does not imply that every requested metric exists.

| Task/dataset | HW/engine | Model | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Natural QA (QASPER / HotpotQA / 2Wiki) | mlx / artifact-recorded hardware | `mlx-community/Qwen3-14B-4bit` | Native Memory | BALANCED | `AVAILABLE_EXISTING` | `AVAILABLE_EXISTING` | `NEEDS_RUN` | `AVAILABLE_EXISTING` | `NEEDS_RUN` |
| Natural QA (QASPER / HotpotQA / 2Wiki) | mlx / artifact-recorded hardware | `mlx-community/Qwen3-32B-4bit` | Native Memory | BALANCED | `AVAILABLE_EXISTING` | `AVAILABLE_EXISTING` | `NEEDS_RUN` | `AVAILABLE_EXISTING` | `NEEDS_RUN` |
| Natural QA (QASPER / HotpotQA / 2Wiki) | mlx / artifact-recorded hardware | `mlx-community/Qwen3-8B-4bit` | Native Memory | BALANCED | `AVAILABLE_EXISTING` | `AVAILABLE_EXISTING` | `NEEDS_RUN` | `AVAILABLE_EXISTING` | `NEEDS_RUN` |
| Exact-identity qualification workload | hf / artifact-recorded hardware | `Qwen/Qwen2.5-1.5B-Instruct` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | hf / artifact-recorded hardware | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Llama-3.1-8B-Instruct-4bit` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Qwen3-4B-4bit` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/gemma-3-1b-it-4bit` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | hf / artifact-recorded hardware | `Qwen/Qwen3-0.6B` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Qwen3-4B-8bit` | Native Memory | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Qwen3-8B-8bit` | Native Memory | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Qwen3-14B-8bit` | Native Memory | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Qwen3-8B-6bit` | Native Memory | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/Llama-3.2-1B-Instruct-8bit` | Native Memory | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | mlx / artifact-recorded hardware | `mlx-community/gemma-3-1b-it-8bit` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |
| Exact-identity qualification workload | hf / artifact-recorded hardware | `Qwen/Qwen2.5-1.5B-Instruct` | Selected Context | BALANCED | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` | `NEEDS_RUN` |

The three MLX natural-QA rows predate immutable bundle resolution: their original-model and generic native-PRA conditions can be normalized, while the Runtime Bundle condition remains `NEEDS_RUN`. Routing-only artifacts remain research diagnostics and do not fill end-task cells.

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
