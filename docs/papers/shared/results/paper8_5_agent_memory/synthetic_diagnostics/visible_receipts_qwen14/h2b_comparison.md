# Frozen replay comparison

Instance `synthetic_versioned_state_convergence`; model `qwen3:14b`; tokenizer `Qwen/Qwen3-14B`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Logical/materialized retention mean | Full/selected/materialized tokens | Logical/materialized saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 1/1 | 100.0% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0% | 354/354/354 | 0 (0.0%)/0 (0.0%) | 0 | 0/0 | 0 | 0 | 0 |
| h2b_verified_write@100%-ceiling;seed=0 | 1/1 | 100.0% | 0.0% | 0.0% | 0.0% | 6/6/6/— | 78.5%/78.5% | 354/278/278 | 76 (21.5%)/76 (21.5%) | 0 | 0/76 | 76 | 0 | 0 |
| h2b_verified_write@100%-ceiling;realization=visible_receipt;seed=0 | 1/1 | 100.0% | 0.0% | 0.0% | 0.0% | 6/6/6/— | 100.0%/133.3% | 354/354/472 | 0 (0.0%)/-118 (-33.3%) | 0 | 0/0 | 0 | 0 | 0 |

`—` means no divergence or no defined denominator.

## Negative-exclusion diagnostics

The false-exclusion value is a narrow immediate proxy: the candidate reacquired a hidden resource when contemporaneous FULL did not. It is not a causal error rate and autonomous reacquisition remains primary.

| Policy | Decisions with exclusion | Cumulative group decisions | Excluded tokens | Immediate reacquisition | Policy-excess proxy |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0 | 0 |
| h2b_verified_write@100%-ceiling;seed=0 | 1 | 1 | 76 | 0 | 0 |
| h2b_verified_write@100%-ceiling;realization=visible_receipt;seed=0 | 0 | 0 | 0 | 0 | 0 |

## Model-visible receipt diagnostics

Receipts preserve the original assistant action and replace only the paired observation. Candidate source tokens, receipt tokens, and actual dropped tokens are reported separately.

| Policy | Decisions with receipts | Receipt count | Candidate tokens | Observation source/receipt/saving tokens | Actually dropped tokens |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0/0/0 | 0 |
| h2b_verified_write@100%-ceiling;seed=0 | 0 | 0 | 76 | 0/0/0 | 76 |
| h2b_verified_write@100%-ceiling;realization=visible_receipt;seed=0 | 1 | 1 | 76 | 15/133/-118 | 0 |
