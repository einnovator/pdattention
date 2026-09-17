# Qwen3 tokenizer geometry audit

The three JSON files apply the common Paper 4.5 original-position mapper to all
22 frozen M2/P1 requests using the locally cached
`mlx-community/Qwen3-4B-4bit` tokenizer on the `.8` Mac. Every request passed
the full-request, selected-plan, mandatory-tail, chat-template, and interval
geometry checks.

| Task | Requests | Mean realized retention | Range | Mean token saving |
|---|---:|---:|---:|---:|
| 3 | 7 | 88.21% | 87.49--88.82% | 11.79% |
| 4 | 6 | 46.89% | 45.26--48.37% | 53.11% |
| 5 | 9 | 49.67% | 44.47--52.35% | 50.33% |

These are engine-tokenizer geometry measurements, not native-K/V or task-
quality results. They establish that the selected record identities can be
mapped without restoring excluded records: retained historical records become
disjoint original-position intervals, while the current causal tail remains
ordinary wire input. A native engine must still demonstrate same-subset
correctness, zero selected-history re-encoding/copy, bounded temporaries, and
lifecycle safety.
