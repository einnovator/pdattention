# AGENTS.md

## Global Objective

Develop a coherent multi-paper research program for Progressive Retrieval Attention (PRA).

## Global Rules

- Keep papers consistent but self-contained.
- Do not fabricate experimental results.
- Mark placeholder numbers as illustrative/mock.
- Prefer clear claims with explicit evidence.
- Keep implementation status separate from future work.
- Reuse shared BibTeX and macros from `docs/shared/`.
- When adding citations, update `docs/shared/references.bib`.
- Preserve a consistent terminology:
  - Progressive Retrieval Attention (PRA)
  - reference handles
  - URI-addressed memory
  - recursive anchors
  - layer-specific memory cache
  - progressive disclosure
  - latent context expansion

## Paper Roadmap

1. Paper 0: Vision and position paper.
2. Paper 1: Standalone PyTorch PRA architecture and benchmark.
3. Paper 2: Hugging Face integration with pretrained models.
4. Paper 3: Runtime/KV-cache/vLLM integration.
5. Paper 4: Scaling laws, theory, and broad comparison.
