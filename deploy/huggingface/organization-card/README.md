---
title: EInnovator
emoji: "🔎"
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---

# EInnovator

EInnovator develops open research and runtime infrastructure for efficient,
addressable context in language models.

## Progressive Retrieval Attention

Progressive Retrieval Attention (PRA) separates logical context, retrieval,
and model-visible execution. Applications can begin with portable Selected
Context and qualify model-specific Native Memory or Native Serving where the
measured quality and economics justify deeper integration.

- [Canonical PRA bundle collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6)
- [Qwen3 0.6B PRA bundle](https://huggingface.co/EInnovator/pra-qwen3-0.6b)
- [Documentation](https://einnovator.github.io/pdattention/)
- [Source code](https://github.com/einnovator/pdattention)

## Publication policy

Repositories named `pra-*` under this organization are the canonical PRA
distribution namespace. A PRA bundle contains structural mappings, optional
learned adapters, profiles, compatibility declarations, and qualification
evidence; it does not duplicate the base model's weights. Production promotion
remains specific to the model revision, tokenizer, engine, profile, workload,
and hardware named by the evidence.

The public PRA Bundles Collection is the discovery layer for current releases.
