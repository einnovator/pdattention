# Autonomous persistent-session evidence

Campaign: `paper85-autonomous-persistent-instruction-epoch-n6-ladder-v1`

| Sequence | Repeat | Strategy | Resolved | Calls | Candidate gross | Paired failure-aware | Call delta | Admissible |
|---|---:|---|---:|---:|---:|---:|---:|---|
| independent-n6-instruction-epoch-ladder-a | 1 | S01_persistent_full/instruction_epoch_n6_ladder_control_v1 | 4/6 | 62 | 0.00% | -- | -- | False |
| independent-n6-instruction-epoch-ladder-a | 1 | S03_instruction_epoch_e1/e1_keep_one_prior_epoch_full_n6_v1 | 5/6 | 70 | 27.90% | 14.21% | 8 | True |
| independent-n6-instruction-epoch-ladder-a | 1 | S03_instruction_epoch_e2/e2_keep_two_prior_epochs_full_n6_v1 | 4/6 | 60 | 16.48% | 0.00% | -2 | True |

`Admissible` describes each row's own trace and grader integrity. The paired
policy comparison is inadmissible because the persistent-FULL control contains
one `patch_apply_failed` submission/grader error. E1 independently misses the
30% saving and no-call-increase gates. E2 loses one FULL success, so its
failure-aware saving is zero even though its raw paired token difference is
21.93%.

## Frozen-prefix diagnosis of the E2 issue-5 loss

The `diagnostics/` directory freezes E2's successful four-issue prefix before
`django__django-16145` and separates first-action sensitivity from autonomous
task quality.

- Five byte-identical FULL requests produce two first-command hashes; five E2
  requests produce one. Temperature zero is not exact trajectory
  determinism on this endpoint.
- E2 retains 11,918 of 15,861 first-request content tokens (24.86% saving).
  Restoring epoch 1 retains 13,945 and reproduces one FULL command exactly;
  restoring epoch 2 retains 13,834 and produces a third command. All are
  semantically equivalent searches for Django's `runserver` implementation,
  so no retired epoch is implicated by the first action.
- Two fresh autonomous E2 forks from the identical prefix and base workspace
  both resolve officially in 13 calls with the identical patch. Each sends
  188,720 tokens rather than its 239,979-token own-trajectory FULL
  counterfactual, a 21.36% gross saving.
- Two autonomous FULL forks from the same prefix both fail official grading,
  in 8 and 18 calls, with different incorrect patches. Mean calls are
  therefore equal (13), while E2 sends 21.92% fewer tokens than the mean FULL
  execution (188,720 versus 241,687).

This targeted reversal does not establish that E2 improves quality. It shows
that the ladder's single E2 loss is not reproducible as a deterministic
selection failure. E2 remains below the 30% saving target and neither policy
is promotable without repeated, interleaved multi-task sequences.
