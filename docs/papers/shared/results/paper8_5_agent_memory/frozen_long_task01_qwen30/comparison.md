# Frozen replay comparison

Instance `django__django-15277`; model `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`; tokenizer `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Retention mean/min/max | Full/selected/materialized tokens | Logical saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 7/7 | 100.0% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 51049/51049/51049 | 0 (0.0%) | 0 | 0/0 | 0 | 0 | 0 |
| safe2_h1_h3@100%-ceiling;Kf=8;Kr=2;seed=0 | 7/7 | 85.7% | 28.6% | 57.1% | 57.1% | 22/22/22/25 | 97.2%/95.8%/100.0% | 51049/49579/49579 | 1470 (2.9%) | 0 | 0/1470 | 1470 | 0 | 1 |
| matched_token_tail@matched-ceiling:678d9bf6;seed=0 | 7/7 | 100.0% | 42.9% | 85.7% | 85.7% | 22/22/22/— | 97.8%/96.7%/100.0% | 51049/49889/49559 | 1160 (2.3%) | 0 | 310/0 | 20 | 0 | 0 |

`—` means no divergence or no defined denominator.

## Negative-exclusion diagnostics

The false-exclusion value is a narrow immediate proxy: the candidate reacquired a hidden resource when contemporaneous FULL did not. It is not a causal error rate and autonomous reacquisition remains primary.

| Policy | Decisions with exclusion | Cumulative group decisions | Excluded tokens | Immediate reacquisition | Policy-excess proxy |
|---|---:|---:|---:|---:|---:|
| FULL | 0 | 0 | 0 | 0 | 0 |
| safe2_h1_h3@100%-ceiling;Kf=8;Kr=2;seed=0 | 5 | 10 | 1470 | 3 | 1 |
| matched_token_tail@matched-ceiling:678d9bf6;seed=0 | 0 | 0 | 0 | 0 | 0 |
