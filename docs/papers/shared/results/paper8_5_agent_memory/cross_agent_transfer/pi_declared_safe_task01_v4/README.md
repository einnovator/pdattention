# Pi declared-safe Task 1 transfer

This folder records the post-fix Pi 0.75.3 transfer on
`django__django-15277` with Qwen3-Coder-30B. Both selective arms use only
whole causal records: an oversized observation is not rewritten unless the
producer declared stable child spans before model-visible materialization.

The paired FULL control resolved in 28 model calls with a 689-byte patch
(`8512d6bc...`). H2/T4 and H2/T8 both resolve officially and reproduce that
exact patch digest, repairing the earlier unsafe-materialization failure.
They do not pass the efficiency gate:

| Arm | Resolved | Calls | Own whole-record saving | Paired input saving | Call delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL | 1 | 28 | 0% | 0% | 0 |
| declared-safe H2/T4 | 1 | 43 | 45.16% | 1.40% | +15 |
| declared-safe H2/T8 | 1 | 39 | 41.53% | 11.81% | +11 |

The H2/T8 trajectory matches the first 13 FULL actions. Selection first
becomes active at request 13, retiring two old causal turns: the broad
`CharField` search and the initial whole-file `expressions.py` view. The next
action diverges after a third schema-invalid `edit` result. Both arms still
solve, but the selective arm performs additional recovery work. This is a
diagnostic one-identity result, not an accuracy estimate or a recommended
profile.

`qualification_t4.json` and `qualification_t8.json` are the authoritative
paired reductions. The older FULL manifest omitted its model-revision field;
the selective manifests contain the observed Ollama digest, so the reducer
honestly reports an identity-field mismatch even though both executions used
the pinned endpoint and model name.
