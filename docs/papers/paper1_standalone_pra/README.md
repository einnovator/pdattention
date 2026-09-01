# Progressive Retrieval Attention: Model-Bounded Sparse Native-KV for Long Context and URI-Addressed Memory

## Focus

Formal model, PyTorch architecture, native-KV routing, bounded long context, datasets,
and controlled synthetic and natural-text experiments.

## Main files

- `paper.tex`
- `AGENTS.md`
- `notes.md`

## Status

Implementation-backed draft with five-seed native-KV, routing, fragmentation,
model-bounded context, residency, and historical adaptation studies. The built
`paper.pdf` is tracked with the project paper sources.

The larger-model dilution receipt is under
docs/papers/shared/results/paper1_standalone_pra/mac_context_dilution/. It
vendors the Qwen3-8B/14B/32B raw rows, generated table, plot, and summary.
The restartable runner and summarizer live on branch
research/paper4-5-runtime under experiments/mac_scaling/; the manifest freezes
example and selected-evidence identity across model sizes.
