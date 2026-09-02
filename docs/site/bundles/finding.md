# Finding a Bundle

Inspect the model and engine first:

```bash
pra inspect Qwen/Qwen3-0.6B -e hf
pra hf list --model Qwen/Qwen3-0.6B
pra hf list --family qwen --engine mlx
pra hf search qwen
pra inspect Qwen/Qwen3-0.6B -e hf -a auto
```

`pra hf list` is the reproducible catalog view. It reads the packaged trusted
registry, works offline, and shows only revision-pinned bundles that may be
considered by `-a auto`. Filter it with `--query`, `--model`, `--family`,
`--engine`, or `--qualification`; use `--json` or `--yaml` for scripts.

`pra hf search [QUERY]` performs a read-only live Hub search without downloading
bundle payloads. It searches `EInnovator` by default:

```bash
pra hf search gemma
pra hf search qwen --author EInnovator --limit 10 --json
pra hf search retrieval --all-authors
```

A live result reports `auto_resolvable: true` only when its repository is also
present in the immutable trusted registry. `--all-authors` can discover
community bundles, but discovery alone never grants trust or makes a repository
eligible for automatic resolution. Select community repositories explicitly
and inspect their manifests before use.

Bare `inspect` downloads only the base-model configuration needed to resolve
its immutable revision. It then reports a compatible published bundle from the
local PRA registry without downloading that bundle. Explicit `-a auto` resolves,
downloads, checksum-validates, and caches the registry-pinned bundle. It never
falls back from an incompatible base revision to a model-name-only match.

`bundle resolve` provides the same explicit resolution operation and reports
the model identity, candidate, trust level, qualification boundary, immutable
Hub revision, local cache path, and selection reason. `auto` considers only the
trusted registry and requires an engine-compatible entry.

An explicit repository remains valid for community or private work:

```bash
pra bundle resolve Qwen/Qwen3-0.6B -e hf \
  -a username/pra-qwen3-0.6b
```

The trust labels mean:

| Label | Meaning |
| --- | --- |
| `eInnovator-qualified` | Checked into the trusted registry with bounded evidence and an immutable revision. |
| `community` | Published by a third party and selected explicitly. |
| `local/private` | Loaded from a local directory or private workflow. |
| `hub-discovered` | Found by live search but not admitted to automatic resolution. |

## Three separate caches

- **Base-model cache:** owned by Hugging Face, Ollama, the GGUF store, or the engine.
- **PRA bundle cache:** Hub repository IDs use the ordinary Hugging Face snapshot cache.
- **Engine artifact cache:** compiled TensorRT engines, OpenVINO conversions, GGUF, and MLX quantizations.

PRA does not create a second copy of the Hugging Face bundle cache.
