# Frozen replay comparison

Instance `django__django-15368`; model `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`; tokenizer `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Retention mean/min/max | Full/selected/materialized tokens | Logical saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 11/11 | 100.0% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 76283/76283/76283 | 0 (0.0%) | 0 | 0/0 | 0 | 0 | 0 |
| safe2_h1_h3@100%-ceiling;Kf=8;Kr=2;seed=0 | 11/11 | 100.0% | 45.5% | 63.6% | 63.6% | 23/23/23/— | 97.3%/96.7%/100.0% | 76283/74213/74213 | 2070 (2.7%) | 0 | 0/2070 | 2070 | 0 | 0 |
| matched_token_tail@matched-ceiling:cfed863a;seed=0 | 11/11 | 100.0% | 36.4% | 72.7% | 72.7% | 23/25/25/— | 97.5%/97.0%/100.0% | 76283/74373/74153 | 1910 (2.5%) | 0 | 160/0 | 60 | 0 | 0 |

`—` means no divergence or no defined denominator.

## Negative-exclusion diagnostics

The false-exclusion value is a narrow immediate proxy: the candidate reacquired a hidden resource when contemporaneous FULL did not. It is not a causal error rate and autonomous reacquisition remains primary.

| Policy | Decisions with exclusion | Cumulative group decisions | Excluded tokens | Immediate reacquisition | Policy-excess proxy |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0 | 0 |
| safe2_h1_h3@100%-ceiling;Kf=8;Kr=2;seed=0 | 10 | 10 | 2070 | 2 | 0 |
| matched_token_tail@matched-ceiling:cfed863a;seed=0 | 0 | 0 | 0 | 0 | 0 |
