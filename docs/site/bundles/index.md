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

The first project-qualified release is the
[`EInnovator/pra-qwen3-0.6b` model card](https://huggingface.co/EInnovator/pra-qwen3-0.6b),
pinned by the registry to Hub commit `25e69076c48a12b5943fe19b0351e68a86ba563e`.
Related releases are grouped in the
[canonical EInnovator PRA Bundles Collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6).
The [EInnovator organization profile](https://huggingface.co/EInnovator)
provides the stable publisher entry point.
The [historical maintainer Collection](https://huggingface.co/collections/jsimao71/pra-bundles-6a97089699497483e5a81c06)
continues to point to the canonical organization release during migration.
Future project bundles are published only under the `EInnovator/pra-*` namespace.
