# Last-14 PRA Adaptation Convergence

This directory contains the final Paper 2 memory-use comparison on frozen
`Qwen/Qwen3-0.6B` with PRA active at layers 14 through 27.

## Protocol

- Train: 12 HotpotQA identities.
- Validation: 4 HotpotQA and 4 QASPER identities.
- Test: 8 new HotpotQA and 8 new QASPER identities.
- Seeds: 11, 23, 37, 53, and 71.
- Updates: 32 oracle-memory steps per trained variant.
- Validation-selected variants: residual width 16 and LoRA rank 4.
- Exact PRA-off parity: passed for logits and greedy generation.
- Native-limit violations: zero.

## Main Result

On HotpotQA, frozen routed PRA gains `+1.901` gold sequence log-probability
over no context. It recovers `11.9%` of direct-evidence benefit and `14.1%`
of feasible full-context benefit while materializing `6.56%` of source K/V.

Residual width 16 is the oracle winner: `+14.478` sequence log-probability,
`90.8%` direct recovery, and `107.3%` full-context recovery at `11.39%`
materialized K/V. It does not improve routed HotpotQA, however. Conditional
LoRA and residual-plus-LoRA are worse under learned routing, so the selected
end-to-end architecture remains frozen last-14 PRA plus the router.

QASPER adaptation gains cannot be converted to context recovery: direct text
lowers cohort likelihood and complete sources exceed the full-context control
budget. These rows remain useful likelihood and decoding sensitivity results.

## Files

- `last14_combo.json`: complete protocol, raw rows, seed summaries, paired effects, and metadata.
- `last14_combo_test.csv`: all 1,120 test method/seed/selection rows.
- `last14_combo_seed_aggregate.csv`: per-seed dataset and selection summaries.
- `last14_combo_aggregate.csv`: five-seed means, dispersion, confidence intervals, and cohort recovery.
- `last14_combo_controls.csv`: no-context, direct-text, and feasible full-context baselines.
- `last14_combo_paired.csv`: combination effects against selected individual mechanisms.
- `recovery_vs_materialized_kv.*`: recovery against physical source-K/V fraction.
- `recovery_vs_adaptation_size.*`: recovery against memory-use parameter fraction.

Adapter checkpoints are excluded from version control; the metrics and plots
are reproducible with `python -m experiments.paper2_hf.qa.run_last14_combo`.
