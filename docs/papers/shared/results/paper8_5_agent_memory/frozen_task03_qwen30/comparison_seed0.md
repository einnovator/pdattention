# Frozen replay comparison

Instance `scikit-learn__scikit-learn-13135`; model `qwen3-coder:30b`; tokenizer `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Retention mean/min/max | Full/selected/materialized tokens | Logical saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 23/23 | 95.7% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 199109/199109/199109 | 0 (0.0%) | 0 | 0/0 | 0 | 0 | 1 |
| dag_certified_exclusion@100%-ceiling;seed=0 | 23/23 | 95.7% | 91.3% | 95.5% | 95.5% | 21/21/21/— | 99.8%/98.2%/100.0% | 199109/198506/198506 | 603 (0.3%) | 0 | 0/603 | 603 | 0 | 1 |
| matched_token_tail@matched-ceiling:35f52f58;seed=0 | 23/23 | 100.0% | 91.3% | 95.5% | 95.5% | 21/21/21/22 | 99.8%/98.3%/100.0% | 199109/198539/198500 | 570 (0.3%) | 0 | 33/0 | 6 | 0 | 0 |
| h1_search_consumed@100%-ceiling;Kf=0;seed=0 | 23/23 | 95.7% | 52.2% | 59.1% | 68.2% | 10/12/13/— | 98.9%/97.9%/100.0% | 199109/196449/196449 | 2660 (1.3%) | 0 | 0/2660 | 2660 | 0 | 1 |
| h1_search_consumed@100%-ceiling;Kf=4;seed=0 | 23/23 | 95.7% | 65.2% | 68.2% | 72.7% | 14/14/16/— | 99.2%/98.0%/100.0% | 199109/197209/197209 | 1900 (1.0%) | 0 | 0/1900 | 1900 | 0 | 1 |
| h2a_write_current_read@100%-ceiling;seed=0 | 23/23 | 95.7% | 73.9% | 77.3% | 77.3% | 17/17/17/20 | 99.7%/98.9%/100.0% | 199109/198276/198276 | 833 (0.4%) | 0 | 0/833 | 833 | 0 | 1 |
| h2b_verified_write@100%-ceiling;seed=0 | 23/23 | 95.7% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 199109/199109/199109 | 0 (0.0%) | 0 | 0/0 | 0 | 0 | 1 |
| h3_read_superseded@100%-ceiling;Kr=1;seed=0 | 23/23 | 95.7% | 91.3% | 95.5% | 95.5% | 21/21/21/— | 99.8%/98.2%/100.0% | 199109/198506/198506 | 603 (0.3%) | 0 | 0/603 | 603 | 0 | 1 |
| h4_working_set@100%-ceiling;Kx=2;seed=0 | 23/23 | 100.0% | 43.5% | 54.5% | 59.1% | 9/9/9/22 | 95.1%/91.1%/100.0% | 199109/187739/187739 | 11370 (5.7%) | 0 | 0/11370 | 11370 | 0 | 0 |
| safe2_h1_h3@100%-ceiling;Kf=0;Kr=1;seed=0 | 23/23 | 100.0% | 47.8% | 54.5% | 59.1% | 10/12/12/22 | 98.6%/96.5%/100.0% | 199109/195846/195846 | 3263 (1.6%) | 0 | 0/3263 | 3263 | 0 | 0 |
| h2a_h3@100%-ceiling;Kr=1;seed=0 | 23/23 | 95.7% | 73.9% | 77.3% | 77.3% | 17/17/17/20 | 99.4%/97.2%/100.0% | 199109/197673/197673 | 1436 (0.7%) | 0 | 0/1436 | 1436 | 0 | 1 |
| all_h1_h2a_h2b_h3_h4@100%-ceiling;Kf=0;Kr=1;Kx=4;seed=0 | 23/23 | 100.0% | 52.2% | 59.1% | 63.6% | 10/12/12/22 | 98.3%/95.5%/100.0% | 199109/195013/195013 | 4096 (2.1%) | 0 | 0/4096 | 4096 | 0 | 0 |

`—` means no divergence or no defined denominator.

## Negative-exclusion diagnostics

The false-exclusion value is a narrow immediate proxy: the candidate reacquired a hidden resource when contemporaneous FULL did not. It is not a causal error rate and autonomous reacquisition remains primary.

| Policy | Decisions with exclusion | Cumulative group decisions | Excluded tokens | Immediate reacquisition | Policy-excess proxy |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0 | 0 |
| dag_certified_exclusion@100%-ceiling;seed=0 | 0 | 0 | 0 | 0 | 0 |
| matched_token_tail@matched-ceiling:35f52f58;seed=0 | 0 | 0 | 0 | 0 | 0 |
| h1_search_consumed@100%-ceiling;Kf=0;seed=0 | 14 | 14 | 2660 | 6 | 1 |
| h1_search_consumed@100%-ceiling;Kf=4;seed=0 | 10 | 10 | 1900 | 5 | 1 |
| h2a_write_current_read@100%-ceiling;seed=0 | 7 | 7 | 833 | 2 | 1 |
| h2b_verified_write@100%-ceiling;seed=0 | 0 | 0 | 0 | 0 | 0 |
| h3_read_superseded@100%-ceiling;Kr=1;seed=0 | 3 | 3 | 603 | 1 | 1 |
| h4_working_set@100%-ceiling;Kx=2;seed=0 | 15 | 15 | 11370 | 0 | 0 |
| safe2_h1_h3@100%-ceiling;Kf=0;Kr=1;seed=0 | 14 | 17 | 3263 | 6 | 1 |
| h2a_h3@100%-ceiling;Kr=1;seed=0 | 7 | 10 | 1436 | 3 | 2 |
| all_h1_h2a_h2b_h3_h4@100%-ceiling;Kf=0;Kr=1;Kx=4;seed=0 | 14 | 24 | 4096 | 6 | 1 |
