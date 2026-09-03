# PRA Runtime Bundle Catalog

PRA Runtime Bundles package structural mappings, profiles, optional learned components, exact compatibility metadata, and qualification evidence. They do not replace or duplicate model weights.

| Order | Runtime bundle | Exact model identity | Role | Evidence | Recommendation | Release |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | [`EInnovator/pra-qwen3-14b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-4bit) | `mlx-community/Qwen3-14B-4bit` | Flagship Native Memory bundle | ENGINE_QUALIFIED | Native Memory with BALANCED | PUBLISHED |
| 2 | [`EInnovator/pra-qwen3-32b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-32b-mlx-4bit) | `mlx-community/Qwen3-32B-4bit` | Large-model Native Memory qualification | ENGINE_QUALIFIED | Native Memory with BALANCED | PUBLISHED |
| 3 | [`EInnovator/pra-qwen3-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-4bit) | `mlx-community/Qwen3-8B-4bit` | Mid-scale Native Memory qualification | ENGINE_QUALIFIED | Native Memory with BALANCED | PUBLISHED |
| 4 | [`EInnovator/pra-qwen2-5-1-5b-instruct`](https://huggingface.co/EInnovator/pra-qwen2-5-1-5b-instruct) | `Qwen/Qwen2.5-1.5B-Instruct` | General instruction-tuned HF routing qualification | CONTROLLED | Selected Context with BALANCED | PUBLISHED |
| 5 | [`EInnovator/pra-qwen2-5-coder-1-5b-instruct`](https://huggingface.co/EInnovator/pra-qwen2-5-coder-1-5b-instruct) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Code-and-instruction-tuned HF routing qualification | CONTROLLED | Selected Context with BALANCED | PUBLISHED |
| 6 | [`EInnovator/pra-llama3-1-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-llama3-1-8b-mlx-4bit) | `mlx-community/Llama-3.1-8B-Instruct-4bit` | Cross-family structural and routing qualification | CONTROLLED | Selected Context with BALANCED | PUBLISHED |
| 7 | [`EInnovator/pra-qwen3-4b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-4bit) | `mlx-community/Qwen3-4B-4bit` | Practical Qwen structural and routing qualification | CONTROLLED | Selected Context with BALANCED | PUBLISHED |
| 8 | [`EInnovator/pra-gemma3-1b-mlx-4bit`](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-4bit) | `mlx-community/gemma-3-1b-it-4bit` | Mixed/sliding-attention structural qualification | CONTROLLED | Selected Context with BALANCED | PUBLISHED |
| 9 | [`EInnovator/pra-qwen3-0.6b`](https://huggingface.co/EInnovator/pra-qwen3-0.6b) | `Qwen/Qwen3-0.6B` | Research/reference mechanism bundle | RESEARCH | Selected Context with BALANCED | PUBLISHED |

The order reflects useful measured evidence, not publication date. `AVAILABLE`, `QUALIFIED`, and `RECOMMENDED` are independent states.
