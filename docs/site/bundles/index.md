# Models and PRA Runtime Bundles

A PRA deployment combines three independently versioned artifacts:

```text
base model weights + PRA Runtime Bundle + engine realization
```

The base model owns language-model weights and tokenizer assets. A PRA Runtime
Bundle packages the model-specific structural map, canonical profiles, optional
learned components, compatibility declarations, checksums, and qualification
evidence. An engine realization is a converted or compiled form such as MLX,
GGUF, OpenVINO, or TensorRT. The bundle does not duplicate either artifact.

Start with the [ordered catalog](catalog.md), then check the exact model,
engine, profile, and mode in the [qualification matrix](qualification-matrix.md).

```bash
pra hf list --family qwen
pra hf search qwen
pra inspect mlx-community/Qwen3-14B-4bit -e mlx -a auto
pra serve mlx-community/Qwen3-14B-4bit -e mlx -a auto -p balanced
```

`pra hf list` reads the offline revision-pinned trust registry. `pra hf search`
queries live Hub metadata. A search result is not trusted or auto-resolvable
unless it is also in the registry. `-a none` disables bundle-specific learned
components, not PRA's training-free Selected Context path.

## Three different statuses

- **AVAILABLE** means an implementation path exists.
- **QUALIFIED** means a stated evidence gate passed for one exact identity.
- **RECOMMENDED** means that qualified path is the current deployment default.

These states are intentionally independent. Native Memory can be available but
not recommended, as measured for AirLLM. Reduced consumer profiles can also be
available while remaining `CALIBRATION_PENDING`.

## What PRA is not

PRA is not a replacement model, a generic LoRA quality fine-tune, or ordinary
external RAG. Native integration is not guaranteed to be faster on every
engine, and deeper integration is not automatically better. Public cards lead
with paired quality and economic measurements, not isolated router recall.

## Distribution contract

Each materially different checkpoint, revision, tokenizer, quantization, and
engine mapping receives an explicit qualification identity. Hugging Face
Collections provide discovery; runtime resolution uses a direct repository ID
and immutable commit.

Project releases live in the
[EInnovator PRA Bundles Collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6).
The [EInnovator organization profile](https://huggingface.co/EInnovator) is the
canonical publisher. Community and local bundles use the same integrity schema
but are never silently promoted to EInnovator-qualified status.
