# Paper 2 Overnight Conditional-LoRA Sweep

## Decision

The validation-only Pareto rule selected `lora_o_r32_s64_lr1`: rank 32,
64 updates (2x the prior budget), learning rate
1e-03, and 1,376,256 trainable parameters
(0.2309% of Qwen3-0.6B). It is packaged for research use,
but it is **not the SDK default**.

The adapter improves clean oracle integration but is brittle to learned-routing errors. On the
untouched HotpotQA test identities, oracle recovery reaches 83.9%
of direct-text benefit and 99.2% of full-context benefit;
oracle F1 is 0.310. Routed recovery is
-54.2%, routed delta-logP is
-8.640, and F1 is
0.022. Frozen PRA remains positive at
11.9% routed direct recovery.

QASPER routed delta-logP rises to +4.967
and F1 to 0.160, but its direct-text denominator is negative, so no recovery
ratio is reported. EM is zero for every routed finalist.

Adding residual-32 does not resolve the mismatch. Its HotpotQA routed delta-logP is
-8.057; paired combo-minus-LoRA effects
change direction across seeds for both datasets and both selection modes.

## Questions Answered

- **Did longer training help?** Only selectively. Rank 32 improves from the 32-update screen to
  64 updates, then slips at 128. Rank 4 and rank 8 become worse at 64 updates under the baseline
  learning rate. The prior result was not uniformly under-converged.
- **Did larger rank help?** Yes for oracle integration: five-seed validation rises from rank 16
  (9.808) to rank 32
  (10.666). It does not help routed HotpotQA.
- **Where did performance saturate?** The tested oracle frontier peaks at rank 32 and 64 updates;
  128 updates regress. Rank 64 was not expanded because the rank-32 routed safety result already
  failed the product gate, so saturation beyond rank 32 is not claimed.
- **Gold rank and F1?** Oracle HotpotQA mean gold rank improves to
  9.65 and F1 to 0.310; routed rank
  worsens to 124.92 and F1 remains near zero.
- **PRA-off exactness?** Yes: all 43 candidate checks are
  exact and native-limit violations are zero.
- **SDK default?** Keep frozen PRA plus the learned router. The packaged LoRA is opt-in for
  oracle/controlled studies, not general routed inference.

## Protocol

- Train: 12 HotpotQA identities, offsets 0--11.
- Validation: four identities per dataset at offset 12; screening and finalist selection use no
  test identities.
- Test: eight identities per dataset at offset 16, loaded only after finalist selection.
- Stage A: ranks 4/8/16/32, 32 and 64 updates, baseline learning rate.
- Stage B: best three ranks, 64/128 updates and 0.5x/1x/2x learning rate.
- Stage C: three rank-diverse finalists over seeds 11, 23, 37, 53, and 71.
- Combination: the selected rank-32 LoRA plus residual-32, once, over the same five seeds.
- Full-context greedy decoding is omitted because 2,048-token eager generation exceeds the
  4-GiB evaluation GPU. Full-context teacher-forced logP and recovery remain measured.

## Files

- `overnight_lora_manifest.json`: predeclared grid and separation rules.
- `validation_ranking.csv`: complete Stage-A/B screen.
- `finalist_validation.csv`: five-seed held-out finalists.
- `test_finalists.csv` and `test_five_seed.csv`: aggregate and per-seed test metrics.
- `paired_vs_frozen.csv` and `combo_paired.csv`: paired effects.
- `recovery_ratios.csv`: direct/full recovery where denominators are valid.
- `pra_off_exactness.csv`: hard retrofit gate.
- `lora_parameter_pareto.pdf`: validation quality against trainable parameter fraction.
- `overnight_lora_sweep.json`: complete raw rows and provenance.
