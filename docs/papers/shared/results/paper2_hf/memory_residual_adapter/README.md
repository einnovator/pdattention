# PRA memory calibration

Five-seed frozen-backbone comparison of unmodified PRA, scalar/per-layer gates,
and late-four-layer residual adapters with bottlenecks 16, 32, and 64.

- Model: `Qwen/Qwen3-0.6B-Base`
- Optimization seeds: 11, 23, 37, 53, 71
- Training: 8 HotpotQA identities, 24 oracle-memory updates
- Held out: 4 HotpotQA and 4 QASPER identities
- Frozen: transformer, LM head, and router
- Best compact setting: residual bottleneck 32, 393,344 parameters (0.066%)
- Routed log-probability delta: HotpotQA `+1.210 +/- 0.248`, QASPER
  `+0.837 +/- 0.241` nats per token
- Disabled-PRA parity: exact in all runs
- Native-limit violations: 0

`memory_gate.json` contains the complete protocol, training traces, raw rows, and
aggregates. CSV files expose per-example, per-seed, and final aggregates. The
`memory_residual_adapter` figure is the paper-ready likelihood/F1 comparison.

The likelihood result is positive, but greedy decoding is not solved: EM remains
zero, HotpotQA F1 changes only slightly, and QASPER F1 declines.
