# AGENTS-Paper1-Standalone

## Mission
Turn the standalone PRA implementation into a rigorous systems paper.

## Improve theory
- Formalize reference graphs, cache state, selection operator, expansion operator, lifecycle and runtime state.
- Expand mathematical notation and complexity analysis.

## Experiments
- Parameter-matched baselines.
- >=5 seeds.
- Oracle, irrelevant, empty and shuffled references.
- Cache-size, alpha, threshold and top-k ablations.
- Cross-attention vs KV injection.
- Frozen vs LoRA vs adapters vs joint finetuning.

## Diagnostics
- Reference selection accuracy.
- Attention heatmaps.
- Layer utilization.
- Cache reuse.
- Latency decomposition.
- GPU memory.
- Failure cases.

## Presentation
- Better diagrams.
- Pseudocode.
- Hyperparameter tables.
- Runtime pipeline.
- Discussion of limitations, safety, cache poisoning and batching.
