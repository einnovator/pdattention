# QASPER Completion and Decision Diagnostic

This follow-up replays the frozen Paper 2 test identities with 32 generated
tokens. It does not modify or replace the frozen 152-comparison behavioral
judge package.

The run uses Qwen3-0.6B, the established last-14 PRA band, the shipped learned
router, 128 direct prompt tokens, 32-token routing parents, top-3 parents, and a
512-token physical memory budget. Adapter rows aggregate seeds 11, 23, 37, 53,
and 71. Train, validation, and test identities are disjoint.

## Main Findings

- Frozen routed PRA produces a valid yes/no prefix on every QASPER item but
  reaches the correct polarity on 25.0% of the eight test identities.
- Frozen oracle memory raises polarity accuracy to 62.5%, locating part of the
  problem in evidence selection.
- The existing residual-16 adapter reaches 72.5% polarity accuracy, 82.5%
  answer containment, and 95.0% EOS completion across 40 seed-item trials.
- Rank-32 LoRA reaches the 32-token cap in 97.5% of trials despite a positive
  teacher-forced polarity margin. Better likelihood is not sufficient evidence
  of better decoding.
- A two-parameter polarity calibration reaches 50.0%; router-margin gating
  selects full memory and does not improve frozen PRA.
- Residual-16 trained directly on routed QASPER memory reaches 100% EOS and F1
  0.511, but only 55.0% polarity accuracy. It is a useful readout diagnostic,
  not a better general adapter.
- The HotpotQA taxonomy remains dominated by relation-near misses under
  residual-16. Iterative relation closure remains Paper 2.5 scope.

The 48-token option was not run: frozen PRA and residual-16 hit the 32-token cap
in only 12.5% and 5.0% of QASPER trials. The widespread cap behavior is
specific to rank-32 LoRA and is itself the diagnosed EOS failure.

`generation_error_analysis.json` is the canonical artifact. CSV files flatten
the per-generation rows, summary metrics, taxonomy, frozen 8-vs-32-token
comparison, calibration data, and validation gate selection. The PDF and PNG
show QASPER polarity accuracy beside length-termination rate.
