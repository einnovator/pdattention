# Frozen replay comparison

Instance `synthetic_versioned_state_convergence`; model `qwen3:14b`; tokenizer `Qwen/Qwen3-14B`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Logical/materialized retention mean | Full/selected/materialized tokens | Logical/materialized saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 1/1 | 100.0% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0% | 354/354/354 | 0 (0.0%)/0 (0.0%) | 0 | 0/0 | 0 | 0 | 0 |
| h2b_verified_write@100%-ceiling;realization=observation_receipt;seed=0 | 1/1 | 100.0% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0% | 354/354/354 | 0 (0.0%)/0 (0.0%) | 0 | 0/0 | 0 | 0 | 0 |

`—` means no divergence or no defined denominator.

## Model-visible receipt diagnostics

Receipts preserve the original assistant action and replace only the paired observation. Candidate source tokens, receipt tokens, and actual dropped tokens are reported separately.

| Policy | Decisions with receipts | Receipt count | Candidate tokens | Observation source/receipt/saving tokens | Abstentions | Abstained source/candidate tokens | Actually dropped tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0/0/0 | 0 | 0/0 | 0 |
| h2b_verified_write@100%-ceiling;realization=observation_receipt;seed=0 | 0 | 0 | 76 | 0/0/0 | 1 | 15/46 | 0 |
