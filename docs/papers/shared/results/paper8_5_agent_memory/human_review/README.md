# Paper 8.5 agent histories for review

Each task has two adjacent files:

- `*.history.json` is the complete canonical machine-readable history. It keeps
  every record's full content, causal identity, roles, runtime provenance, and
  DAG annotations.
- `*.history.md` is the economical human/AI review view. Long task text,
  commands, reasoning, and observations use explicit head/tail excerpts with
  omitted line/character counts and the full-content SHA-256.

| Task | Compact review | Complete JSON |
|---|---|---|
| `scikit-learn__scikit-learn-13135` | [Markdown](scikit-learn__scikit-learn-13135.history.md) | [JSON](scikit-learn__scikit-learn-13135.history.json) |

Regenerate one or more task pairs with:

```bash
python -m experiments.paper8_5_agent_memory.export_review_history \
  --trajectory task1.traj.json task2.traj.json \
  --output-directory docs/papers/shared/results/paper8_5_agent_memory/human_review
```

Do not infer behavioral safety from a DAG annotation alone. Only candidates
with `default_exclusion_eligible=true` and a non-null certificate pass the
current operational duplicate proof; policy quality still requires frozen and
autonomous evaluation.
