# Quantized Bundles

PRA bundles are tied to an exact model revision, quantization, and runtime.
An adapter or measurement from a 4-bit checkpoint is not silently reused for
its 6-bit, 8-bit, full-precision, or runtime-quantized counterpart.

## Fleet-fit catalog

| Host class | Exact model | Weight format | PRA bundle | Runtime smoke peak | Current evidence |
| --- | --- | --- | --- | ---: | --- |
| Apple Silicon, 16 GB+ | `mlx-community/gemma-3-1b-it-8bit` | MLX 8-bit, group 64 | [`pra-gemma3-1b-mlx-8bit`](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-8bit) | 1.30 GiB | CONTROLLED; native candidate |
| Apple Silicon, 16 GB+ | `mlx-community/Llama-3.2-1B-Instruct-8bit` | MLX 8-bit, group 64 | [`pra-llama3-2-1b-mlx-8bit`](https://huggingface.co/EInnovator/pra-llama3-2-1b-mlx-8bit) | 1.24 GiB | ENGINE_QUALIFIED |
| Apple Silicon, 16 GB+ | `mlx-community/Qwen3-4B-8bit` | MLX 8-bit, group 64 | [`pra-qwen3-4b-mlx-8bit`](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-8bit) | 4.03 GiB | ENGINE_QUALIFIED; learned router opt-in |
| Apple Silicon, 16 GB+ | `mlx-community/Qwen3-8B-6bit` | MLX 6-bit, group 64 | [`pra-qwen3-8b-mlx-6bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-6bit) | 6.24 GiB | SMOKE |
| Apple Silicon, 24 GB+ | `mlx-community/Qwen3-8B-8bit` | MLX 8-bit, group 64 | [`pra-qwen3-8b-mlx-8bit`](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-8bit) | 8.15 GiB | SMOKE |
| Apple Silicon, 32 GB+ | `mlx-community/Qwen3-14B-8bit` | MLX 8-bit, group 64 | [`pra-qwen3-14b-mlx-8bit`](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-8bit) | 14.66 GiB | SMOKE |
| NVIDIA CUDA, 4 GB+ | `Qwen/Qwen2.5-1.5B-Instruct` | bitsandbytes LLM.int8 | [`pra-qwen2-5-1-5b-instruct-bnb-8bit`](https://huggingface.co/EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit) | 1.71 GiB CUDA | SMOKE |

Peak values are allocator readings from one exact-checkpoint load and bounded
generation smoke. They exclude operating-system and unrelated process memory,
and they are not capacity promises for long prompts or large PRA native-memory
budgets. Leave additional headroom for K/V cache, selected memory, generation,
and concurrent sessions.

## Choosing a format

- **4-bit MLX** remains the capacity-oriented choice and has the strongest
  current PRA end-task evidence for Qwen3 8B, 14B, and 32B.
- **6-bit MLX** is a useful middle point when 8-bit weights constrain context
  or concurrency but 4-bit is unnecessarily aggressive.
- **8-bit MLX** is the quality-oriented local choice when unified memory leaves
  room for the model, ordinary K/V, and PRA-selected K/V.
- **bitsandbytes int8** is the Windows/Linux NVIDIA option in this catalog. It
  quantizes the immutable Hugging Face checkpoint when it is loaded.

These choices describe weight residency, not PRA K/V precision. Quantized PRA
K/V is a separate storage/runtime policy and requires its own quality and
latency calibration.

## Inspect and run

```bash
pra hf search qwen3 --author EInnovator
pra hf pull EInnovator/pra-qwen3-8b-mlx-8bit
pra inspect mlx-community/Qwen3-8B-8bit -e mlx \
  -a EInnovator/pra-qwen3-8b-mlx-8bit
pra serve mlx-community/Qwen3-8B-8bit -e mlx \
  -a EInnovator/pra-qwen3-8b-mlx-8bit -p balanced
```

The new bundles establish exact structural and runtime compatibility. Qwen3-4B
8-bit additionally includes an exact five-seed learned router: it improves
QASPER R@20% by `0.120` but reduces HotpotQA by `0.227`, so it is opt-in and the
generic router remains default. Paired end-task quality, Native Memory parity,
TTFT, ITL, and sustained throughput remain explicitly `NOT_MEASURED`.
