# PRAttention-Centered Survey of Long Context and Retrieval

This package contains an educational review paper that organizes architectures related to Progressive Retrieval Attention (PRAttention) into seven categories:

1. Dense, sparse, and approximate long-context attention
2. Recurrence and compression
3. Differentiable external memory
4. Retrieval-augmented pretraining and generation
5. Attention-level and KV retrieval
6. Structural and hierarchical indexing
7. Agents with search tools

## Files

- `pra_survey.tex` - complete LaTeX source
- `pra_survey.pdf` - compiled review paper
- `AGENTS.md` - instructions for Codex or another coding/research agent to verify, expand, and maintain the paper

## Scope

The paper begins with a multi-page reconstruction of PRAttention Paper 0 and Paper 1, including the prototype architecture, formalism, preliminary WikiText-2 experiments, and counterfactual evaluation doctrine. It then reviews representative approaches by motivation, mechanism, equations, pseudocode/PyTorch sketches, original experiments, research lineage, downstream influence, relation to PRA, and proposed comparative experiments.

It also includes:

- a landscape diagram in a landscape page;
- a cross-architecture comparison table;
- a benchmark and ablation matrix;
- a distinction between research “paper trail” and the 2026 `PaperTrail` claim-evidence interface;
- proposed extensions combining PRA with hierarchical indexes, cache virtualization, provenance, and agentic escalation.

## Build

```bash
latexmk -pdf -interaction=nonstopmode survey.tex
```

Clean auxiliary files:

```bash
latexmk -c
```

## Important status

This is a substantial tutorial draft, not yet a submission-ready systematic review. Before publication, verify every quantitative claim against the original paper, add complete experiment tables for each reviewed architecture, broaden coverage of 2024-2026 KV-compression and agentic retrieval work, and decide whether to split the document into a tutorial paper and a PRA-focused position/experiment-design paper.
