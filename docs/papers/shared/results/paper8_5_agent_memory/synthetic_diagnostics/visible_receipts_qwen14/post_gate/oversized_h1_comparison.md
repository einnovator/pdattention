# Frozen replay comparison

Instance `synthetic_discovery_consumed_oversized_old_search`; model `qwen3:14b`; tokenizer `Qwen/Qwen3-14B`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Logical/materialized retention mean | Full/selected/materialized tokens | Logical/materialized saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 1/1 | 100.0% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0% | 1449/1449/1449 | 0 (0.0%)/0 (0.0%) | 0 | 0/0 | 0 | 0 | 0 |
| h1_all_branches_consumed_strict@100%-ceiling;Kf=0;seed=0 | 1/1 | 100.0% | 0.0% | 0.0% | 0.0% | 7/7/7/— | 23.3%/23.3% | 1449/337/337 | 1112 (76.7%)/1112 (76.7%) | 0 | 0/1112 | 1112 | 0 | 0 |
| h1_all_branches_consumed_strict@100%-ceiling;Kf=0;realization=observation_receipt;seed=0 | 1/1 | 100.0% | 0.0% | 0.0% | 0.0% | 7/7/7/— | 100.0%/28.3% | 1449/1449/410 | 0 (0.0%)/1039 (71.7%) | 0 | 0/0 | 0 | 0 | 0 |

`—` means no divergence or no defined denominator.

## Negative-exclusion diagnostics

The false-exclusion value is a narrow immediate proxy: the candidate reacquired a hidden resource when contemporaneous FULL did not. It is not a causal error rate and autonomous reacquisition remains primary.

| Policy | Decisions with exclusion | Cumulative group decisions | Excluded tokens | Immediate reacquisition | Policy-excess proxy |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0 | 0 |
| h1_all_branches_consumed_strict@100%-ceiling;Kf=0;seed=0 | 1 | 1 | 1112 | 0 | 0 |
| h1_all_branches_consumed_strict@100%-ceiling;Kf=0;realization=observation_receipt;seed=0 | 0 | 0 | 0 | 0 | 0 |

## Model-visible receipt diagnostics

Receipts preserve the original assistant action and replace only the paired observation. Candidate source tokens, receipt tokens, and actual dropped tokens are reported separately.

| Policy | Decisions with receipts | Receipt count | Candidate tokens | Observation source/receipt/saving tokens | Abstentions | Abstained source/candidate tokens | Actually dropped tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0/0/0 | 0 | 0/0 | 0 |
| h1_all_branches_consumed_strict@100%-ceiling;Kf=0;seed=0 | 0 | 0 | 1112 | 0/0/0 | 0 | 0/0 | 1112 |
| h1_all_branches_consumed_strict@100%-ceiling;Kf=0;realization=observation_receipt;seed=0 | 1 | 1 | 1112 | 1082/43/1039 | 0 | 0/0 | 0 |
