# Bundle Qualification Queue

A passing family contract only establishes structural compatibility. Model loading, adapter training, three-condition measurement, qualification, and publication are separate gates.

| Order | Exact base model | Family | Contract | Engines | Load | Adapter | No PRA / no adapter / adapter evidence | Publication |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `Qwen/Qwen3-30B-A3B` | qwen3_moe | attention-only MoE isolation (`CONTRACT_TESTED`) | hf, mlx, vllm | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 2 | `google/gemma-3-12b-it` | gemma3 | mixed full/sliding attention (`CONTRACT_TESTED`) | hf, mlx | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 3 | `google/gemma-3-27b-it` | gemma3 | mixed full/sliding attention (`CONTRACT_TESTED`) | hf, mlx | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 4 | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | mistral3 | nested text decoder (`CONTRACT_TESTED`) | hf, mlx, vllm | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 5 | `openai/gpt-oss-20b` | gpt_oss | full-attention-only MoE with learned sinks (`CONTRACT_TESTED`) | hf, mlx, vllm | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 6 | `Qwen/QwQ-32B` | qwen2 | dense Qwen decoder (`CONTRACT_TESTED`) | hf, mlx, vllm | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 7 | `meta-llama/Llama-3.3-70B-Instruct` | llama3 | dense Llama decoder (`CONTRACT_TESTED`) | hf, mlx, vllm | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

Reduced consumer-layer profiles remain `CALIBRATION_PENDING` until held-out workload-scale quality supports promotion. The queue does not create a catalog entry or a public-bundle claim.
