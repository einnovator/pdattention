# LM-head adaptation control

Five-seed global-readout control following the completed PRA-conditioned
calibration sequence.

- Model: `Qwen/Qwen3-0.6B-Base`
- Optimization seeds: 11, 23, 37, 53, 71
- Training: 8 HotpotQA identities, 24 oracle-memory updates
- Held out: 4 HotpotQA and 4 QASPER identities
- Ordinary-language control: 1,024 fixed WikiText-2 validation tokens
- Frozen: transformer backbone, router, and token embeddings
- Variants: rank-8 LM-head LoRA, untied full LM head, and final norm plus
  untied full LM head
- Full-head optimizer: zero-momentum SGD because stateful full-matrix
  optimizers exceed the 4 GiB test GPU
- No-PRA training control: KL distillation from frozen answer-position logits
- Native-limit violations: 0

The control is negative for the global-readout hypothesis. On HotpotQA,
routed-minus-no-memory gold likelihood is `+0.104 +/- 0.009` for head LoRA and
about `+0.090 +/- 0.026` for either full-head variant, versus `+0.110` for
frozen PRA and `+1.846 +/- 0.088` for PRA-conditional late-band LoRA. Generated
HotpotQA F1 remains `0.056` and EM remains zero.

Unlike conditional adaptation, every global readout changes ordinary behavior.
Head LoRA raises WikiText-2 loss by `+0.0161 +/- 0.0071`; full-head and
norm-plus-head tuning raise it by about `+0.0938 +/- 0.0069`. Adding final
normalization does not measurably improve any reported outcome. QASPER is
retained as a supplementary sensitivity result because direct evidence text is
negative on this four-example cohort.

`lm_head_control.json` contains the complete protocol, traces, raw rows, and
aggregates. CSV files expose per-example, per-seed, language-control, and final
cross-stage comparison data. `lm_head_control.pdf` and `.png` contain the
paper-ready sequence figure.
