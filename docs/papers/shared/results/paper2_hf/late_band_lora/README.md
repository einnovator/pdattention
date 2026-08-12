# Conditional late-band LoRA

Five-seed comparison of frozen PRA, residual bottleneck 32, and
PRA-conditioned attention-output LoRA ranks 2, 4, and 8 in Qwen's final four
decoder layers.

- Model: `Qwen/Qwen3-0.6B-Base`
- Optimization seeds: 11, 23, 37, 53, 71
- Training: 8 HotpotQA identities, 24 oracle-memory updates
- Held out: 4 HotpotQA and 4 QASPER identities
- Frozen: base transformer, LM head, router, and all inactive adapters
- LoRA target: native attention output projection, selected-memory branch only
- LoRA alpha: equal to rank; dropout: zero
- No-PRA control: evaluated on every training update and exactly unchanged
- Disabled-PRA held-out parity: exact in every run
- Native-limit violations: 0

Rank 8 uses 98,304 parameters (0.0165% of the base model) and raises routed
HotpotQA gold likelihood by `+1.846 +/- 0.088` nats per token, versus
`+1.210 +/- 0.248` for the 393,344-parameter residual adapter. The paired
difference favors rank 8 in all five seeds. Routed HotpotQA EM remains zero and
F1 is only `0.071`; the experiment therefore demonstrates parameter-efficient
likelihood calibration, not reliable generation.

QASPER direct text is a negative control on this four-example cohort. Its LoRA
likelihood shifts are reported but do not establish QA utility.

`late_band_lora.json` contains the complete protocol, training traces, raw
rows, and aggregates. CSV files expose per-example, per-seed, and final
aggregates. `late_band_lora.pdf` and `.png` are the paper-ready figure.
