# Long-horizon frozen replay: django-15277

This is engine-independent ordinary-text, next-action evidence over decisions
20--26 of a 26-decision mini-swe-agent trajectory that resolved officially
under plain FULL history. It is not an autonomous policy-success result and
contains no K/V or latency claim.

The frozen consumer is direct `mlx-lm` 0.29.1 on the 48 GB M4 Pro host, using
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` revision
`6e302ea604ad9ab206367e2c501d1571023e7b6d`, its local tokenizer, temperature
zero, top-p one, seed zero, and 1,024 maximum generated tokens. FULL-A is the
contemporaneous reference. FULL-B reproduces all 7/7 contents byte-for-byte
and all 7/7 commands.

[`comparison.md`](comparison.md) contains the reduced results. The semantic
arm applies H1 with `Kf=8` plus H3 with `Kr=2`; on this trajectory only H1
fires. The matched tail uses the semantic arm's per-decision materialized-token
ceiling and allows boundary tool-observation trimming (`threshold=1`).

| Arm | Materialized retention | Exact content | Exact command | Valid action |
|---|---:|---:|---:|---:|
| FULL repeat | 100.00% | 7/7 | 7/7 | 7/7 |
| H1 `Kf=8` + H3 `Kr=2` | 97.12% | 2/7 | 4/7 | 6/7 |
| Matched token tail | 97.08% | 3/7 | 6/7 | 7/7 |

The tail is only 20 cumulative tokens below the semantic arm over 51,049 FULL
tokens. H1 removes two old search-map causal groups from decisions 22--26.
Those groups still contain useful navigation/progress evidence: the semantic
arm has three immediate reacquisition events, one policy-excess reacquisition,
and one format-invalid action. Delaying whole-group retirement does not make
the rule competitive with recency on this task.
