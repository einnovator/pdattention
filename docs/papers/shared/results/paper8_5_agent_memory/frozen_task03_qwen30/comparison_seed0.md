# Frozen replay comparison

Instance `scikit-learn__scikit-learn-13135`; model `qwen3-coder:30b`; tokenizer `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Retention mean/min/max | Full/selected/materialized tokens | Logical saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 23/23 | 95.7% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 199109/199109/199109 | 0 (0.0%) | 0 | 0/0 | 0 | 0 | 1 |
| dag_certified_exclusion@100%-ceiling;seed=0 | 23/23 | 95.7% | 91.3% | 95.5% | 95.5% | 21/21/21/— | 99.8%/98.2%/100.0% | 199109/198506/198506 | 603 (0.3%) | 0 | 0/603 | 603 | 0 | 1 |
| matched_token_tail@matched-ceiling:35f52f58;seed=0 | 23/23 | 100.0% | 91.3% | 95.5% | 95.5% | 21/21/21/22 | 99.8%/98.3%/100.0% | 199109/198539/198500 | 570 (0.3%) | 0 | 33/0 | 6 | 0 | 0 |
| h1_search_consumed@100%-ceiling;seed=0 | 23/23 | 95.7% | 52.2% | 59.1% | 68.2% | 10/12/13/— | 98.9%/97.9%/100.0% | 199109/196449/196449 | 2660 (1.3%) | 0 | 0/2660 | 2660 | 0 | 1 |

`—` means no divergence or no defined denominator.

## Negative-exclusion diagnostics

The false-exclusion value is a narrow immediate proxy: the candidate reacquired a hidden resource when contemporaneous FULL did not. It is not a causal error rate and autonomous reacquisition remains primary.

| Policy | Decisions with exclusion | Cumulative group decisions | Excluded tokens | Immediate reacquisition | Policy-excess proxy |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0 | 0 |
| dag_certified_exclusion@100%-ceiling;seed=0 | 0 | 0 | 0 | 0 | 0 |
| matched_token_tail@matched-ceiling:35f52f58;seed=0 | 0 | 0 | 0 | 0 | 0 |
| h1_search_consumed@100%-ceiling;seed=0 | 14 | 14 | 2660 | 6 | 1 |
