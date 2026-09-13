# Long-horizon frozen replay: django-15368

This directory measures decisions 20--30 of a 30-decision mini-swe-agent
trajectory that resolved officially under plain FULL history. The evidence is
frozen ordinary-text next-action behavior, not autonomous policy success and
not physical K/V/runtime performance.

The original Ollama/Qwen3-Coder FULL replay first reproduced all 30 historical
contents and commands. An identical late FULL repeat then reproduced only 4/11
contents and 7/11 commands, so that endpoint failed the selection-attribution
gate. Those two artifacts are retained as a negative control.

The qualified comparison instead uses direct `mlx-lm` 0.29.1 on the 48 GB M4
Pro host with `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` revision
`6e302ea604ad9ab206367e2c501d1571023e7b6d`, its exact tokenizer, temperature
zero, top-p one, seed zero, and 1,024 maximum generated tokens. MLX FULL-B
reproduces all 11/11 FULL-A contents and commands; decision 21 also reproduces
in three independent FULL calls.

[`comparison.md`](comparison.md) compares FULL, H1 `Kf=8` + H3 `Kr=2`, and a
near-exact matched token tail. H1 abstains on this trajectory, so the semantic
treatment isolates H3. The tail uses the semantic arm's per-decision ceiling
and permits boundary tool-observation trimming (`threshold=1`).

| Arm | Materialized retention | Exact content | Exact command | Valid action |
|---|---:|---:|---:|---:|
| FULL repeat | 100.00% | 11/11 | 11/11 | 11/11 |
| H1 `Kf=8` + H3 `Kr=2` | 97.29% | 5/11 | 7/11 | 11/11 |
| Matched token tail | 97.21% | 4/11 | 8/11 | 11/11 |

The matched tail is 60 cumulative tokens below the semantic arm over 76,283
FULL tokens. H3 removes one 207-token older read of
`django/db/models/query.py` from decisions 21--30. It first changes the command
at decision 23; the token tail first changes a command at decision 25. H3
therefore does not outperform blind recency at a practically identical token
count on this task.
