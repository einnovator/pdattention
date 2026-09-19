# Post-fix MLX autonomous agent gate

This bundle reruns the admitted `django__django-15277` identity after the
engine-correctness changes at source revision `02e53d01`.  It uses
Qwen2.5-Coder-14B-Instruct-4bit at temperature zero on the 48 GB M4 Pro host.
The 16 GB M5 attempt was stopped after it became swap-bound and is not included
as scientific evidence.

| Arm | Official | Calls | Physical / logical input | Own saving | First divergence | Re-encoded history | Selected-K/V copy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain | 1/1 | 6 | n.a. | n.a. | reference | n.a. | n.a. |
| PRA-100, task-aware | 1/1 | 6 | 8,920 / 8,920 | 0.00% | none | 0 | 0 B |
| PRA-90, matched causal token-tail | 0/1 | 9 | 24,246 / 27,069 | 10.43% | action 3 | 0 | 0 B |
| PRA-90, task-aware floors | 1/1 | 6 | 8,725 / 8,920 | 2.19% | action 6 | 0 | 0 B |

PRA-100 reproduces all six assistant actions and the submitted patch exactly.
This qualifies the live resident-K/V mechanism for this model/profile.  The
90% causal-tail policy then fails despite correct sparse consumption: it loses
the efficient edit path, repeats failed edits, and submits deletion of the
target file.  The structural policy retaining recent, source, progress,
mutation, and verification state succeeds in six calls and submits the exact
plain patch, but this short trajectory offers only 2.19% cumulative physical
input saving.

The result separates the two questions cleanly.  MLX correctness passes; a
nominal 90% token ceiling is not itself a safe agent-memory policy.  The large
`total_kv_copy_bytes` and consumer-temporary values are reported separately
from the zero selected-history re-encoding and zero selected-K/V copy.  They
remain workload-economics targets rather than being mislabeled as logical
selection failure.

The source campaign expects mini-swe-agent 2.4.0, whereas this local engine
gate used the already qualified 2.4.6 harness.  Every official-result receipt
records that configuration difference.
