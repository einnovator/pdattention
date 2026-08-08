# Progressive Retrieval Attention

Progressive Retrieval Attention (PRA) is a research prototype for decoder-only
Transformers that resolve explicit reference handles into external, layer-specific memory.
Instead of placing every referenced token in the prompt, the model routes with compact
chunk gists and materializes detailed K/V only for selected references.

```text
prompt reference handle
  -> URI resolver
  -> reference chunks
  -> layer-specific routing gists and token K/V
  -> row-local reference and chunk selection
  -> memory attention
```

## Explore the project

- [Getting started](getting-started.md) covers installation, CLI use, tests, and site builds.
- [Architecture](architecture.md) explains the prompt, resolver, cache, routing, and
  attention path.
- [API reference](api/index.md) is generated from the Python source and docstrings.

## Main packages

| Package | Responsibility |
| --- | --- |
| `common` | Reusable configuration, training, metrics, logging, plotting, and checkpoints |
| `pra_core` | Framework-neutral reference handles, tables, and dataset records |
| `data` | Dataset generation, tokenization, collation, and datamodules |
| `pra_torch` | PyTorch PRA model, memory cache, routing, resolution, training, and evaluation |
| `hf_wrappers` | Experimental compatibility adapter for Hugging Face decoder models |

!!! note "Research status"
    This codebase is an experimental research system. Architecture variants and reference
    interventions should be compared with controlled datasets and multiple seeds before
    drawing model-quality conclusions.
