# Frozen replay comparison

Instance `scikit-learn__scikit-learn-13135`; model `qwen3-coder:30b`; tokenizer `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Retention mean/min/max | Full/selected/materialized tokens | Logical saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 23/23 | 95.7% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 199109/199109/199109 | 0 (0.0%) | 0 | 0/0 | 0 | 0 | 1 |
| dag_certified_exclusion@100%-ceiling | 23/23 | 95.7% | 91.3% | 95.5% | 95.5% | 21/21/21/— | 99.8%/98.2%/100.0% | 199109/198506/198506 | 603 (0.3%) | 0 | 0/603 | 603 | 0 | 1 |
| middle_recency@matched-ceiling:35f52f58 | 23/23 | 100.0% | 87.0% | 90.9% | 90.9% | 21/21/21/22 | 97.1%/77.0%/100.0% | 199109/191309/191309 | 7800 (3.9%) | 0 | 0/7197 | 7197 | 0 | 0 |
| middle_recency@90%-floor | 23/23 | 95.7% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0%/100.0% | 199109/199109/199109 | 0 (0.0%) | 1807 | 19899/0 | 0 | 0 | 1 |
| middle_lexical@90%-floor | 23/23 | 100.0% | 30.4% | 45.5% | 59.1% | 6/6/6/22 | 94.7%/90.1%/100.0% | 199109/187323/187323 | 11786 (5.9%) | 1807 | 8113/0 | 0 | 0 | 0 |
| spine_plus_lexical@90%-floor | 23/23 | 95.7% | 39.1% | 45.5% | 59.1% | 8/8/8/9 | 95.5%/90.1%/100.0% | 199109/188706/188706 | 10403 (5.2%) | 4030 | 9496/0 | 0 | 0 | 1 |
| dag_certified_plus_lexical@90%-floor | 23/23 | 95.7% | 39.1% | 45.5% | 59.1% | 8/8/8/9 | 95.7%/90.1%/100.0% | 199109/189263/189263 | 9846 (4.9%) | 4030 | 10053/0 | 0 | 0 | 1 |

`—` means no divergence or no defined denominator.
