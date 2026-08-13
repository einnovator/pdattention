# Bounded Memory Computation Survey

This package contains a survey of bounded memory computation for language models. It uses
nine computational questions to compare seven literature families and treats Progressive
Retrieval Attention (PRAttention) as a worked synthesis rather than a presumed winner.

## Literature families

1. Dense, sparse, and approximate long-context attention
2. Recurrence, state-space models, and compression
3. Differentiable external memory
4. Retrieval-augmented pretraining and generation
5. Attention-level and KV retrieval, retention, and reuse
6. Structural and hierarchical indexing
7. Agents with search tools

## Files

- `pra_survey.tex` - canonical LaTeX source
- `pra_expanded.tex` - PRA synthesis and evidence boundary
- `approaches_expanded.tex` - method-level literature atlas
- `references.bib` - primary-source bibliography
- `survey.tex` - compatibility build wrapper
- `pra_survey.pdf` and `survey.pdf` - compiled papers

## Scope

The paper distinguishes logical availability, causal activity, discovery, search geometry,
associative closure, materialized detail, model depth, decisions/behavior, and serving cost.
It reviews representative approaches by mechanism, evidence, limitations, relation to PRA,
and matched comparative experiments. The PRA chapter synthesizes the current native-KV,
bounded-context, positional-geometry, pretrained-retrofit, and behavioral evidence.

It also includes:

- explicit claim labels for implemented, measured, and proposed mechanisms;
- a landscape diagram and cross-architecture comparison tables;
- a taxonomy for storage, search, compression, materialization, iteration, retraining, and
  active bounds;
- a causal benchmark and ablation matrix;
- an architecture ladder separating measured one-shot PRA from proposed associative closure
  and later smart materialization.

## Build

```bash
latexmk -pdf -interaction=nonstopmode survey.tex
latexmk -pdf -interaction=nonstopmode pra_survey.tex
```

Clean auxiliary files:

```bash
latexmk -c
```

## Status

This is a narrative architecture survey, not a systematic review or meta-analysis. Measured
PRA results come from the accompanying paper series and repository artifacts; proposed
associative-closure and smart-materialization mechanisms are deliberately identified as
future work.
