# Frozen request-4 add-back diagnostic

This directory diagnoses the first action difference in the corrected MLX
Scikit-learn-14496 E2+F1C run.  It is a frozen, ordinary-text next-action
experiment, not an autonomous task-quality result and not a native-K/V speed
measurement.  The candidate request digest is reconstructed exactly from the
autonomous trace.  All comparisons use the same Qwen3-Coder-30B model,
tokenizer, temperature zero, active-task prefix, and five-issue completed
prefix.

## Epoch screen

`epoch_screen.json` contains two repeated FULL controls, two repeated E2+F1C
controls, and one add-back for each retired source epoch.

| Arm | Valid action | Added tokens over candidate | Materialized tokens | Saving vs FULL request |
|---|---:|---:|---:|---:|
| FULL, repeats 1--2 | 2/2 | 10,643 | 26,506 | 0.00% |
| E2+F1C, repeats 1--2 | 0/2 | 0 | 15,863 | 40.15% |
| Restore epoch 1 | 1/1 | 1,720 | 17,583 | 33.66% |
| Restore epoch 2 | 1/1 | 6,489 | 22,352 | 15.67% |
| Restore epoch 3 | 1/1 | 1,310 | 17,173 | 35.21% |

Both FULL controls produce the identical valid command.  Both candidate
controls reach the 1,024-token completion ceiling without a valid command.
Every whole-epoch restoration recovers a valid action, but the commands are
not identical.  This is stable selection sensitivity, not evidence for one
uniquely necessary fact.

## Previous Scikit-learn epoch decomposition

`epoch3_groups.json` restores each of the five causal groups from the previous
Scikit-learn issue separately.  Every group restores a valid action while
retaining 38.63--39.67% request-level materialized-token saving:

| Causal group | Prior action type | Added tokens | Restored active-task action |
|---|---|---:|---|
| `turn-000036` | verify mutation | 129 | direct edit |
| `turn-000035` | mutate source | 205 | direct edit |
| `turn-000032` | find source file | 220 | targeted grep |
| `turn-000033` | grep source | 351 | targeted grep |
| `turn-000034` | read source span | 405 | direct edit |

The non-unique recovery rejects a narrow interpretation that a particular
mutation or source fact was required.

## Unrelated-group control

`unrelated_groups.json` restores one small group from each of two unrelated
older repositories.  The FULL and candidate controls reproduce the preceding
valid/invalid contrast.  An 82-token SymPy group restores a targeted grep; a
118-token Django group restores a direct edit.  Therefore recovery does not
identify a Scikit-learn resource edge, mutation fact, or verification fact.
It is a generic prompt-conditioning effect.

The newly captured candidate response enters a repetitive
`grep -v "def|class|import|..."` sequence and never closes the required command
block, even though the endpoint reports `finish_reason=stop`.  A tiny unrelated
causal group prevents that degeneration.  This explains why blindly adding a
generic M1 floor is not a sound policy: an arbitrary perturbation can repair
one frozen next action without preserving autonomous efficiency.

The packed ordinary-text candidate fails here, whereas the original-position
native sparse-K/V E2+F1C run emits a valid fourth action and resolves the task.
Selection and consumption mode must therefore remain separate factors.  This
oracle supplies no basis for adding a DAG edge or changing E2+F1C; it instead
requires an autonomous policy comparison under the intended consumer.

## Guardrails

- Add-backs use evaluator-hidden source-epoch identity only for diagnosis.
- No add-back result enters an accuracy--saving curve.
- Packed ordinary-text behavior is compared only within this oracle.  It is
  not treated as exact action parity with original-position native sparse K/V.
- Autonomous qualification remains necessary for any resulting policy.
- Runner revisions: `cfee831d` for the epoch screen, `872e7256` for the epoch-3
  decomposition, and `ec64d1bf` for the unrelated-group control with exact
  response capture.
