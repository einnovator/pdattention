# Paper 3.1: Generated Summaries as Retrieval Indices

This directory contains the Paper 3.1 source and built PDF. Measured artifacts
live in `docs/papers/shared/results/paper3_1_summary_index/`; runnable protocols
live in `experiments/paper3_1_summary_index/`.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_3_1.tex
```

The paper must keep three quantities separate throughout:

1. Persistent summary-index bytes.
2. Backing source/native-KV storage.
3. Native K/V physically materialized after routing.

Generated summaries are lossy addresses. They are not native-K/V compression,
replacement evidence, or a summary-only answer context.

## Measured status

The frozen standalone and multi-index studies are complete. The paper has been
editorially consolidated and marked ready for external review; mechanisms and
numerical claims should remain frozen unless review identifies a validity
error. Validation selected
one summary policy per dataset, then a separate two-identity-per-dataset split
froze the small RRF/fusion grid. The multi-index replay evaluated 24 held-out
identities at matched `K={2,4,8}`. Summaries recovered unique QASPER and
MuSiQue evidence parents, but adding the summary to `L+E+QK` produced no
resolved positive marginal effect under any tested admission family. LoRA and
downstream answer-generation gates therefore remain closed.

The main paper now distinguishes discovery from consumption and compares the
addressability question with reversible context compression, prompt
compression, agent memory, programmatic state, RAG, and KV retrieval. This is a
taxonomy and claims-boundary comparison, not a direct benchmark against those
systems.

Regenerate publication tables and plots with:

```powershell
python experiments/paper3_1_summary_index/build_publication_artifacts.py
python experiments/paper3_1_summary_index/build_multi_index_artifacts.py
```
