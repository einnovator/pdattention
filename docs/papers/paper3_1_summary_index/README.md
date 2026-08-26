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

The frozen study is complete for this iteration. Validation selected one policy
per dataset and held-out evaluation reported those policies without test-time
reselection. The only positive point estimate was QASPER (`+0.1625` recall
versus native mean), with a wide interval; HotpotQA tied, and 2Wiki and MuSiQue
regressed. The teacher did not establish general headroom, so LoRA distillation
and downstream answer-generation gates remained closed.

Regenerate publication tables and plots with:

```powershell
python experiments/paper3_1_summary_index/build_publication_artifacts.py
```
