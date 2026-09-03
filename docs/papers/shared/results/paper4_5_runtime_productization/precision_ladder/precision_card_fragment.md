## Precision qualification

A precision row qualifies only the exact conversion, engine, mode, profile, and linked evidence.

| Model | Size | Family | Precision/encoding | Engine | Mode | Profile | Qualification | Datasets |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `mlx-community/Qwen3-14B-4bit` | 14B | qwen3 | INT4 / MLX-4bit | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |
| `mlx-community/Qwen3-14B-8bit` | 14B | qwen3 | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |
| `mlx-community/Qwen3-4B-4bit` | 4B | qwen3 | INT4 / MLX-4bit | mlx | Selected Context | BALANCED | CONTROLLED | qasper, hotpotqa, multihop-rag |
| `mlx-community/Qwen3-4B-8bit` | 4B | qwen3 | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | 1B | llama3 | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |
| `mlx-community/gemma-3-1b-it-4bit` | 1B | gemma3 | INT4 / MLX-4bit | mlx | Selected Context | BALANCED | CONTROLLED | qasper, hotpotqa |
| `mlx-community/gemma-3-1b-it-8bit` | 1B | gemma3 | INT8 / MLX-8bit | mlx | Selected Context | BALANCED | CONTROLLED | qasper, hotpotqa, 2wikimultihopqa |
