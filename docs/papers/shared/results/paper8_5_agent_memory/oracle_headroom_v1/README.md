# Paper 8.5 bounded omission oracle

Evidence class: offline next-action oracle; not deployable selector quality and
not autonomous task accuracy.

The cohort fixes one late decision in three previously repeat-qualified
Qwen3-Coder-30B trajectories. Each `single_*.json` removes one complete middle
causal group from FULL. `*_queue.json` files bind pair/beam candidates to the
task, decision, and FULL-reference digest. `subset_*.json` files are executed
counterfactuals. System/task/head/tail records are never eligible.

| Directory | Instance | Decision | FULL tokens | Singleton trials |
|---|---|---:|---:|---:|
| `task01_d24` | `django__django-15277` | 24 | 7,599 | 20 |
| `task02_d25` | `django__django-15368` | 25 | 6,597 | 21 |
| `task03_d20` | `scikit-learn__scikit-learn-13135` | 20 | 11,111 | 16 |

`oracle_trials.csv` is the trial-level ledger. `oracle_summary.csv` and
`oracle_summary.json` reduce by task and omission depth. Exact command and
semantic transition are separate outcomes. Semantic transition means a valid
action with the same typed operation and target-resource set; it is weaker than
exact replay and does not establish eventual task success.

The legacy `task01_d24/subset_depth2-001.json` is the exact-command pair trial
created before qualification-specific filenames were added. Later artifacts
encode `exact_command` or `semantic_transition` in the filename so the two
tracks cannot overwrite one another.

