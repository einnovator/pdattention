# Models and PRA Bundles

A PRA deployment combines three independently versioned artifacts:

```text
Hugging Face base model
        +
PRA bundle: structural map + optional learned adapters + profiles + evidence
        +
engine realization: HF, MLX, vLLM, llama.cpp, or another runtime
```

The **base model** owns language-model weights and tokenizer assets. A **PRA
bundle** contains only the model-specific PRA mapping, optional learned PRA
components, profile policy, compatibility declarations, checksums, and
qualification evidence. An **engine artifact** is a converted or compiled form
such as GGUF, an OpenVINO model, an MLX quantization, or a TensorRT engine. A PRA
bundle never duplicates base-model weights or treats an engine artifact as an
adapter.

Use one base model and zero or one bundle:

```bash
pra inspect Qwen/Qwen3-0.6B -e hf -a auto
pra serve Qwen/Qwen3-0.6B -e hf -a auto -p balanced
```

`-a none` disables bundle-specific adapters, not PRA itself. Selected Context
and generic structural support remain available.

## Distribution contract

Each materially different base-model/checkpoint variant uses a separate Hub
repository. Immutable revisions and release tags evolve one compatible bundle.
Hugging Face Collections group related repositories for discovery; runtime
resolution always uses a direct repository ID plus immutable commit.

Automatic selection uses only entries in the checked-in trusted registry.
Explicit local and community sources remain loadable, but they are never
silently promoted to project-qualified status.

## Published collection

Project-qualified releases are grouped in the
[canonical EInnovator PRA Bundles Collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6).
The [EInnovator organization profile](https://huggingface.co/EInnovator)
provides the stable publisher entry point.

| Exact base identity | Canonical bundle | Default routing | Learned-router evidence |
| --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | [`EInnovator/pra-qwen3-0.6b`](https://huggingface.co/EInnovator/pra-qwen3-0.6b) | Existing QASPER router | First controlled public proof |
| `mlx-community/Qwen3-4B-4bit` | [`EInnovator/pra-qwen3-4b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-4bit) | Generic cosine | QASPER `R@20%`: `0.413 -> 0.640`; HotpotQA: `0.386 -> 0.342` |
| `mlx-community/Qwen3-14B-4bit` | [`EInnovator/pra-qwen3-14b-mlx-4bit`](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-4bit) | Generic cosine | QASPER `R@20%`: `0.318 -> 0.679`; HotpotQA: `0.494 -> 0.314` |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | [`EInnovator/pra-llama3-1-8b-mlx-4bit`](https://huggingface.co/EInnovator/pra-llama3-1-8b-mlx-4bit) | Generic cosine | QASPER `R@20%`: `0.318 -> 0.468`; HotpotQA: `0.616 -> 0.421` |
| `mlx-community/gemma-3-1b-it-4bit` | [`EInnovator/pra-gemma3-1b-mlx-4bit`](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-4bit) | Generic cosine | QASPER `R@20%`: `0.226 -> 0.454`; HotpotQA: `0.336 -> 0.318` |

The four MLX bundles compare the same PRA feature path with and without an
asymmetric learned router. Because every learned router helps QASPER but none
improves HotpotQA recall, `balanced` keeps generic routing and
`qasper-learned` is an explicit dataset-qualified profile. The evidence applies
only to the pinned 4-bit MLX revision shown in each model card.

The [historical maintainer Collection](https://huggingface.co/collections/jsimao71/pra-bundles-6a97089699497483e5a81c06)
continues to point to the canonical organization release during migration.
Future project bundles are published only under the `EInnovator/pra-*` namespace.
