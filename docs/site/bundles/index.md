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
