# Frozen replay comparison

Instance `scikit-learn__scikit-learn-13135`; model `qwen3-coder:30b`; tokenizer `/Users/admin.jorge.simao/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit/snapshots/6e302ea604ad9ab206367e2c501d1571023e7b6d`. Valid-action and exact-content rates use attempted decisions; command rates use decisions where FULL emitted a command. All comparisons are against contemporaneous FULL.

| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Logical/materialized retention mean | Full/selected/materialized tokens | Logical/materialized saving | Mandatory overflow | Whole-turn over/under target | Unused matched budget | Transport failures | Format-invalid |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| FULL | 23/23 | 95.7% | 100.0% | 100.0% | 100.0% | —/—/—/— | 100.0%/100.0% | 199109/199109/199109 | 0 (0.0%)/0 (0.0%) | 0 | 0/0 | 0 | 0 | 1 |
| FULL;materialization=tool_structured_evidence;threshold=512 | 8/8 | 100.0% | 25.0% | 50.0% | 50.0% | 3/3/3/— | 100.0%/92.4% | 57480/57480/53145 | 0 (0.0%)/4335 (7.5%) | 0 | 0/0 | 0 | 0 | 0 |
| FULL;materialization=tool_matched_span;threshold=512 | 8/8 | 100.0% | 0.0% | 25.0% | 25.0% | 3/3/3/— | 100.0%/71.2% | 57480/57480/41232 | 0 (0.0%)/16248 (28.3%) | 0 | 0/0 | 0 | 0 | 0 |
| FULL;materialization=tool_head_tail;threshold=512 | 8/8 | 100.0% | 0.0% | 12.5% | 12.5% | 3/3/3/— | 100.0%/69.1% | 57480/57480/40074 | 0 (0.0%)/17406 (30.3%) | 0 | 0/0 | 0 | 0 | 0 |

`—` means no divergence or no defined denominator.
