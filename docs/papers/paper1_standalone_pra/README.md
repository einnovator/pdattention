# Progressive Retrieval Attention: Bounded Sparse Native-KV for Addressable Logical Memory

## Focus

Formal model, PyTorch architecture, native-KV routing, bounded long context, datasets,
and controlled synthetic and natural-text experiments.

## Main files

- `paper.tex` and `paper.pdf`: canonical audited manuscript for publication review
- `paper_report.tex` and `paper_report.pdf`: long technical-report backup
- `AGENTS.md`
- `notes.md`

## Status

Implementation-backed manuscript with five-seed native-K/V transport, fragmentation,
sparse scaling, bounded-context, routing-cost, and residency studies, plus a separate
pretrained MLX calibration. The built `paper.pdf` is tracked with the paper sources.

The compact experiment map is in `paper.tex`. Run
`python scripts/audit_paper1_headlines.py` from the repository root to verify the headline
numbers against frozen artifacts; the generated receipt is `NUMERICAL_AUDIT.json`.

The report backup preserves the earlier comprehensive implementation narrative and
line-numbered reimplementation guide. It is retained for reference, but it is not the
canonical manuscript for publication review.

The pretrained calibration receipt is under
docs/papers/shared/results/paper1_standalone_pra/mac_context_dilution/. It
vendors the Qwen3-8B/14B/32B raw rows, generated table, plot, and summary.
The restartable runner and summarizer live on branch
research/paper4-5-runtime under experiments/mac_scaling/; the manifest freezes
example and selected-evidence identity across model sizes.
