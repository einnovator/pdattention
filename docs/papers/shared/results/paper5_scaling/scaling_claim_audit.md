# Paper 5 Scaling Claim Audit

## Supported by this branch

- **Measured:** five-seed controlled routing from 32,768 to 8,388,608 logical tokens.
- **Measured:** exact and coarse-to-fine retrieval, active native-K/V tokens, layer-token K/V, index bytes, search latency, routing-only concurrency, and CPU/CUDA hardware slices.
- **Measured:** fixed-difficulty evidence recall remained in [1.000, 1.000] at 128 active tokens.
- **Measured:** evidence-dispersion sweeps vary regions and chain depth separately from logical address-space size.
- **Measured:** hard-negative difficulty exposes adaptive escalation: the hard condition fails without a confidence threshold and recovers under thresholded budget expansion.
- **Inherited calibration only:** Paper 4 Frozen, LoRA, full-weight, and PRA-native controlled consumer-learning results.

## Not supported yet

- No claim of infinite context, bounded total runtime, or production ANN efficiency.
- No matched Gemma 270M/1B/4B quality curve and no smaller-PRA-versus-larger-native claim.
- No end-to-end NLL, perplexity, EM/F1, TTFT, TPOT, generation throughput, dollar cost, or production HBM curve.
- No Apple Silicon run and no 4B+ run. These remain preregistered ladder cells.
- Analytical native/RAG/PRA state rows are accounting baselines, not measured quality or latency.

## Interpretation rule

The controlled result tests whether selected physical attention state can track a fixed task working set while the addressable pool grows. GPU-resident routing-index bytes still grow with memory, and the current Python coarse-to-fine path can be slower than exact GPU GEMM. Both are scaling costs, not exceptions to hide.
